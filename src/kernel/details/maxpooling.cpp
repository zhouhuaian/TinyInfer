#include "maxpooling.hpp"
#include "kernel/abstract/kernel_factory.hpp"
#include "runtime/runtime_graph.hpp"
#include "status_code.hpp"
#include <cstddef>
#include <cstdint>
#include <glog/logging.h>

namespace TinyInfer {

MaxPooling::MaxPooling(uint32_t padding_h, uint32_t padding_w,
                       uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h,
                       uint32_t stride_w)
    : NoAttrKernel("MaxPooling"), padding_h_(padding_h), padding_w_(padding_w),
      kernel_h_(kernel_h), kernel_w_(kernel_w), stride_h_(stride_h),
      stride_w_(stride_w) {}

InferStatus MaxPooling::Forward(const std::vector<sftensor> &inputs,
                                std::vector<sftensor> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array is empty";
    return InferStatus::InferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  CHECK(stride_h_ > 0 && stride_w_ > 0) << "Stride must greater than 0";

  const uint32_t batch = inputs.size();
  const uint32_t kernel_h = kernel_h_;
  const uint32_t kernel_w = kernel_w_;

#pragma omp parallel for num_threads(batch)
  for (uint32_t b = 0; b < batch; ++b) {
    const sftensor &input = inputs.at(b);
    sftensor &output = outputs.at(b);

    CHECK(input != nullptr && !input->empty())
        << "The " << b << " input tensor is empty";

    sftensor input_;
    // 对输入Tensor进行扩充
    if (padding_h_ > 0 || padding_w_ > 0) {
      input_ = Pad(input, {padding_h_, padding_h_, padding_w_, padding_w_},
                   std::numeric_limits<float>::lowest());
    } else {
      input_ = input;
    }

    const uint32_t input_c = input_->channels();
    const uint32_t input_h = input_->rows();
    const uint32_t input_w = input_->cols();

    uint32_t output_h =
        uint32_t(std::floor((input_h - kernel_h) / stride_h_ + 1));
    uint32_t output_w =
        uint32_t(std::floor((input_w - kernel_w) / stride_w_ + 1));

    CHECK(output_h > 0 && output_w > 0) << "The " << b << " output shape error";

    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The " << b << " output tensor is empty";
      output = std::make_shared<ftensor>(input_c, output_h, output_w);
    }

    CHECK(output->channels() == input_c && output->rows() == output_h &&
          output->cols() == output_w)
        << "The " << b << " output tensor dimension is wrong";

    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat &in_channel = input_->slice(ic);
      arma::fmat &out_channel = output->slice(ic);
      // 在输入Tensor上滑动
      for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w_) {
        for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h_) {
          // out_col_ptr是指向输出Tensor的列指针
          float *out_col_ptr = out_channel.colptr(int(c / stride_w_));
          // 扫描池化窗口，计算其中最大值
          float max_val = std::numeric_limits<float>::lowest();
          for (uint32_t w = 0; w < kernel_w; ++w) {
            const float *in_col_ptr = in_channel.colptr(c + w) + r;
            for (uint32_t h = 0; h < kernel_h; ++h) {
              float val = *(in_col_ptr + h);
              max_val = (val > max_val) ? val : max_val;
            }
          }
          *(out_col_ptr + int(r / stride_h_)) = max_val;
        }
      }
    }
  }
  return InferStatus::InferSuccess;
}

ParseParamAttrStatus MaxPooling::GetInstance(const srunop &op,
                                             skernel &maxpooling) {
  if (op == nullptr) {
    LOG(ERROR) << "Operator is empty";
    return ParseParamAttrStatus::OpEmpty;
  }

  const auto &params = op->params;

  if (params.find("stride") == params.end()) {
    LOG(ERROR) << "Stride parameter is missing";
    return ParseParamAttrStatus::ParamMissingStride;
  }

  const auto &stride_param =
      dynamic_cast<RuntimeParamIntArr *>(params.at("stride"));
  if (stride_param == nullptr) {
    LOG(ERROR) << "Stride parameter is missing";
    return ParseParamAttrStatus::ParamMissingStride;
  }

  if (params.find("padding") == params.end()) {
    LOG(ERROR) << "Padding parameter is missing";
    return ParseParamAttrStatus::ParamMissingPadding;
  }

  const auto &padding_param =
      dynamic_cast<RuntimeParamIntArr *>(params.at("padding"));
  if (padding_param == nullptr) {
    LOG(ERROR) << "Padding parameter is missing";
    return ParseParamAttrStatus::ParamMissingPadding;
  }

  if (params.find("kernel_size") == params.end()) {
    LOG(ERROR) << "Kernel size parameter is missing";
    return ParseParamAttrStatus::ParamMissingKernelSize;
  }

  const auto &kernel_param =
      dynamic_cast<RuntimeParamIntArr *>(params.at("kernel_size"));
  if (kernel_param == nullptr) {
    LOG(ERROR) << "Kernel size parameter is missing";
    return ParseParamAttrStatus::ParamMissingKernelSize;
  }

  const uint32_t dims = 2;
  const auto &padding = padding_param->value;
  const auto &stride = stride_param->value;
  const auto &kernel_size = kernel_param->value;

  if (padding.size() != dims) {
    LOG(ERROR) << "Padding parameter is wrong";
    return ParseParamAttrStatus::ParamMissingPadding;
  }

  if (stride.size() != dims) {
    LOG(ERROR) << "Stride parameter is wrong";
    return ParseParamAttrStatus::ParamMissingStride;
  }

  if (kernel_size.size() != dims) {
    LOG(ERROR) << "Kernel size parameter is wrong";
    return ParseParamAttrStatus::ParamMissingKernelSize;
  }

  maxpooling = std::make_shared<MaxPooling>(
      padding.at(0), padding.at(1), kernel_size.at(0), kernel_size.at(1),
      stride.at(0), stride.at(1));

  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

KernelRegisterWrapper MaxPoolingGetInstance("nn.MaxPool2d",
                                            MaxPooling::GetInstance);

} // namespace TinyInfer