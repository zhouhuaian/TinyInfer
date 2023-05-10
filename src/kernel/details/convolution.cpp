#include "convolution.hpp"
#include "kernel/abstract/kernel_factory.hpp"
#include "runtime/runtime_graph.hpp"
#include "runtime/runtime_param.hpp"
#include "status_code.hpp"
#include "tick.hpp"
#include <algorithm>
#include <cstdint>
#include <glog/logging.h>
#include <memory>
#include <utility>
#include <vector>

namespace TinyInfer {

Convolution::Convolution(uint32_t out_channels, uint32_t in_channels,
                         uint32_t kernel_h, uint32_t kernel_w,
                         uint32_t padding_h, uint32_t padding_w,
                         uint32_t stride_h, uint32_t stride_w, uint32_t groups,
                         bool use_bias)
    : AttrKernel("Convolution"), padding_h_(padding_h), padding_w_(padding_w),
      stride_h_(stride_h), stride_w_(stride_w), groups_(groups),
      use_bias_(use_bias) {

  in_channels /= groups_;

  this->InitWeights(out_channels, in_channels, kernel_h, kernel_w);

  if (use_bias_) {
    this->InitBias(out_channels, 1, 1, 1);
  }
}

InferStatus Convolution::Forward(const std::vector<sftensor> &inputs,
                                 std::vector<sftensor> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "Input tensor array empty";
    return InferStatus::InferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "Input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  CHECK(stride_h_ > 0 && stride_w_ > 0) << "Stride must greater than 0";

  CHECK(!this->weights_.empty()) << "Weight count must greater than 0";

  if (this->use_bias_) {
    CHECK(this->bias_.size() == this->weights_.size())
        << "Weight and bias count do not match";
  }

  const uint32_t kernel_ct = this->weights_.size(); // 卷积核kernel数目
  CHECK(kernel_ct > 0 && kernel_ct % groups_ == 0) << "Kernel count error";

  // !
  // 注意：对于分组卷积，kernel的通道数已经和分组后的输入特征图通道数保持一致了！
  const uint32_t kernel_c = this->weights_.at(0)->channels();
  const uint32_t kernel_h = this->weights_.at(0)->rows();
  const uint32_t kernel_w = this->weights_.at(0)->cols();
  CHECK(kernel_c > 0 && kernel_h > 0 && kernel_w > 0) << "Kernel shape error";

  for (uint32_t k = 0; k < kernel_ct; ++k) {
    const auto &kernel = this->weights_.at(k);
    CHECK(kernel->channels() == kernel_c && kernel->rows() == kernel_h &&
          kernel->cols() == kernel_w)
        << k << " kernel shape error";

    if (this->use_bias_) {
      const auto &bias = this->bias_.at(k);
      CHECK(bias != nullptr && !bias->empty()) << k << " bias empty";
    }
  }

  const uint32_t gkernel_ct = kernel_ct / groups_; // 每组的kernel数目
  const uint32_t plane = kernel_h * kernel_w; // kernel单个通道内的元素数

  std::vector<std::shared_ptr<arma::frowvec>> kernel_rows(
      kernel_ct); // 保存单分组下所有展平的kernels
  std::vector<std::vector<std::shared_ptr<arma::frowvec>>> gkernel_rows_vec(
      groups_, // 保存多分组下所有展平的kernels
      std::vector<std::shared_ptr<arma::frowvec>>(gkernel_ct));

  // 展平kernels
  // 单分组
  if (groups_ == 1) {
    for (uint32_t k = 0; k < kernel_ct; ++k) {
      const auto &kernel = this->weights_.at(k);
      // ! 直接复用kernel的内存，不必开辟新空间保存展平后的kernel
      kernel_rows.at(k) = std::make_shared<arma::frowvec>(
          const_cast<float *>(kernel->raw_ptr()), kernel_c * plane, false,
          true);
    }
  }
  // 多分组
  else {
    for (uint32_t g = 0; g < groups_; ++g) {
      std::vector<std::shared_ptr<arma::frowvec>> gkernel_rows(
          gkernel_ct); // 保存一个分组内展平的kernels
      for (uint32_t k = 0; k < gkernel_ct; ++k) {
        const auto &kernel = this->weights_.at(g * gkernel_ct + k);
        gkernel_rows.at(k) = std::make_shared<arma::frowvec>(
            const_cast<float *>(kernel->raw_ptr()), kernel_c * plane, false,
            true);
      }
      gkernel_rows_vec.at(g) = std::move(gkernel_rows);
    }
  }

  const uint32_t batch = inputs.size();

#pragma omp parallel for num_threads(batch)
  for (uint32_t b = 0; b < batch; ++b) {
    const auto &input = inputs.at(b);
    CHECK(input != nullptr && !input->empty()) << b << " input tensor empty";

    // 扩充输入特征图
    sftensor input_;
    if (padding_h_ > 0 || padding_w_ > 0) {
      input_ =
          Pad(input, {padding_h_, padding_h_, padding_w_, padding_w_}, 0.f);
    } else {
      input_ = input;
    }

    const uint32_t input_c = input_->channels();
    const uint32_t input_w = input_->cols();
    const uint32_t input_h = input_->rows();
    CHECK(input_c % groups_ == 0) << b << " input tensor channel error";

    uint32_t ginput_c = input_c / groups_; // 每组的特征图通道数
    // !
    // 这里也说明了分组卷积情况下，kernel的通道数已经和分组后特征图通道数一致！
    CHECK(ginput_c == kernel_c)
        << b << " input tensor grouped channel not equal to kernel channel";

    const uint32_t output_h =
        uint32_t(std::floor((input_h - kernel_h) / stride_h_ + 1));
    const uint32_t output_w =
        uint32_t(std::floor((input_w - kernel_w) / stride_w_ + 1));
    CHECK(output_h > 0 && output_w > 0) << b << " output shape error";

    auto &output = outputs.at(b);
    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << b << " output tensor empty";
      output = std::make_shared<ftensor>(kernel_ct, output_h, output_w);
    }
    CHECK(output->channels() == kernel_ct && output->rows() == output_h &&
          output->cols() == output_w)
        << b << " output tensor shape error";

    uint32_t out_plane = output_h * output_w; // 输出特征图单个通道内元素数目

    // 分组进行im2col和gemm
    for (uint32_t g = 0; g < groups_; ++g) {
      // 展平该组内的特征图通道
      arma::fmat in_mat(ginput_c * plane, out_plane); // 保存im2col后的特征图
      for (uint32_t ic = 0; ic < ginput_c; ++ic) {
        const auto &in_channel =
            input_->slice(g * ginput_c + ic); // 特征图单个通道
        int in_mat_c_idx = 0;                 // in_mat的列号
        // 自左而右、自上而下滑动
        for (uint32_t w = 0; w < input_w - kernel_w + 1; w += stride_w_) {
          for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h_) {
            float *in_mat_c_ptr =
                in_mat.colptr(in_mat_c_idx) + ic * plane; // in_mat的列指针
            in_mat_c_idx += 1;
            // 按列展平一个kernel窗口内的元素
            for (uint32_t kw = 0; kw < kernel_w; ++kw) {
              const float *window_ptr =
                  in_channel.colptr(w + kw) + r; // 窗口内的列指针
              memcpy(in_mat_c_ptr, window_ptr, kernel_h * sizeof(float));
              in_mat_c_ptr += kernel_h;
            }
          }
        }
      }

      // 执行该组的矩阵乘法
#pragma omp parallel for schedule(dynamic)
      for (uint32_t k = 0; k < gkernel_ct; ++k) {
        arma::fmat out_channel(output->slice(g * gkernel_ct + k).memptr(),
                               output_h, output_w, false, true);

        if (groups_ == 1) {
          out_channel = *kernel_rows.at(k) * in_mat;
        } else {
          out_channel = *gkernel_rows_vec.at(g).at(k) * in_mat;
        }
        CHECK(out_channel.size() == out_plane);

        // 加上偏置
        if (this->use_bias_) {
          const auto &bias = this->bias_.at(g * gkernel_ct + k);
          out_channel += bias->index(0);
        }
      }
    }
  }

  return InferStatus::InferSuccess;
}

ParseParamAttrStatus Convolution::Creator(const srunop &op,
                                          skernel &convolution) {
  if (!op) {
    LOG(ERROR) << "Operator is empty";
    return ParseParamAttrStatus::OpEmpty;
  }

  const auto &params = op->params;

  if (params.find("in_channels") == params.end()) {
    LOG(ERROR) << "In channels parameter missing";
    return ParseParamAttrStatus::ParamMissingInChannels;
  }
  const auto &in_channels =
      dynamic_cast<RuntimeParamInt *>(params.at("in_channels"));
  if (!in_channels) {
    LOG(ERROR) << "In channels parameter missing";
    return ParseParamAttrStatus::ParamMissingInChannels;
  }

  if (params.find("out_channels") == params.end()) {
    LOG(ERROR) << "Out channels parameter missing";
    return ParseParamAttrStatus::ParamMissingOutChannels;
  }
  const auto &out_channels =
      dynamic_cast<RuntimeParamInt *>(params.at("out_channels"));
  if (!out_channels) {
    LOG(ERROR) << "Out channels parameter missing";
    return ParseParamAttrStatus::ParamMissingOutChannels;
  }

  if (params.find("kernel_size") == params.end()) {
    LOG(ERROR) << "Kernel size parameter missing";
    return ParseParamAttrStatus::ParamMissingKernelSize;
  }
  const auto &kernel_size =
      dynamic_cast<RuntimeParamIntArr *>(params.at("kernel_size"));
  if (!kernel_size) {
    LOG(ERROR) << "Kernel size parameter missing";
    return ParseParamAttrStatus::ParamMissingKernelSize;
  }

  if (params.find("stride") == params.end()) {
    LOG(ERROR) << "Stride parameter missing";
    return ParseParamAttrStatus::ParamMissingStride;
  }
  const auto &stride = dynamic_cast<RuntimeParamIntArr *>(params.at("stride"));
  if (!stride) {
    LOG(ERROR) << "Stride parameter missing";
    return ParseParamAttrStatus::ParamMissingStride;
  }

  if (params.find("padding") == params.end()) {
    LOG(ERROR) << "Padding parameter missing";
    return ParseParamAttrStatus::ParamMissingPadding;
  }
  const auto &padding =
      dynamic_cast<RuntimeParamIntArr *>(params.at("padding"));
  if (!padding) {
    LOG(ERROR) << "Padding parameter missing";
    return ParseParamAttrStatus::ParamMissingPadding;
  }

  if (params.find("padding_mode") == params.end()) {
    LOG(ERROR) << "Padding mode parameter missing";
    return ParseParamAttrStatus::ParamMissingPaddingMode;
  }
  const auto &padding_mode =
      dynamic_cast<RuntimeParamStr *>(params.at("padding_mode"));
  if (!padding_mode || padding_mode->value != "zeros") {
    LOG(ERROR) << "Padding mode parameter missing or unsupported";
    return ParseParamAttrStatus::ParamMissingPaddingMode;
  }

  if (params.find("dilation") == params.end()) {
    LOG(ERROR) << "Dilation parameter missing";
    return ParseParamAttrStatus::ParamMissingDilation;
  }
  const auto &dilation =
      dynamic_cast<RuntimeParamIntArr *>(params.at("dilation"));
  if (!dilation || dilation->value.size() != 2) {
    LOG(ERROR) << "Dilation parameter missing";
    return ParseParamAttrStatus::ParamMissingDilation;
  }

  CHECK(dilation->value.at(0) != 1 || dilation->value.at(1))
      << "Unsupported dilation value";

  if (params.find("bias") == params.end()) {
    LOG(ERROR) << "Bias parameter missing";
    return ParseParamAttrStatus::ParamMissingBias;
  }
  const auto &bias = dynamic_cast<RuntimeParamBool *>(params.at("bias"));
  if (!bias) {
    LOG(ERROR) << "Bias parameter missing";
    return ParseParamAttrStatus::ParamMissingBias;
  }

  if (params.find("groups") == params.end()) {
    LOG(ERROR) << "Groups parameter missing";
    return ParseParamAttrStatus::ParamMissingGroups;
  }
  const auto &groups = dynamic_cast<RuntimeParamInt *>(params.at("groups"));
  if (!groups) {
    LOG(ERROR) << "Groups parameter missing";
    return ParseParamAttrStatus::ParamMissingGroups;
  }

  const uint32_t dims = 2;
  const auto &kernel_size_val = kernel_size->value;
  const auto &padding_val = padding->value;
  const auto &stride_val = stride->value;

  CHECK(kernel_size_val.size() == dims) << "Kernel size parameter size error";

  CHECK(padding_val.size() == dims) << "Padding parameter size error";

  CHECK(stride_val.size() == dims) << "Stride parameter size error";

  // 创建convolution kernel
  convolution = std::make_shared<Convolution>(
      out_channels->value, in_channels->value, kernel_size_val.at(0),
      kernel_size_val.at(1), padding_val.at(0), padding_val.at(1),
      stride_val.at(0), stride_val.at(1), groups->value, bias->value);

  // 加载权重
  const auto &attrs = op->attrs;

  if (attrs.find("weight") == attrs.end()) {
    LOG(ERROR) << "Weight attribute missing";
    return ParseParamAttrStatus::AttrMissingWeight;
  }
  const auto &weight = attrs.at("weight");
  CHECK(!weight->shape.empty()) << "Weight attribute shape error";

  const std::vector<float> &weight_vals = weight->get<float>();
  convolution->set_weights(weight_vals);

  // 加载偏置
  if (bias->value) {
    if (attrs.find("bias") == attrs.end()) {
      LOG(ERROR) << "Bias attribute missing";
      return ParseParamAttrStatus::AttrMissingBias;
    }
    const auto &bias = attrs.at("bias");
    CHECK(!bias->shape.empty() && bias->shape.at(0) == out_channels->value)
        << "Bias attribute shape error";

    const std::vector<float> &bias_vals = bias->get<float>();
    convolution->set_bias(bias_vals);
  }

  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

KernelRegisterWrapper ConvCreator("nn.Conv2d", Convolution::Creator);

} // namespace TinyInfer
