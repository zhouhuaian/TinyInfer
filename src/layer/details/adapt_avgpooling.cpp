#include "adapt_avgpooling.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "status_code.hpp"
#include <glog/logging.h>

namespace TinyInfer {

AdaptAvgPooling::AdaptAvgPooling(uint32_t output_h, uint32_t output_w)
    : NoAttrLayer("AdaptAvgPooling"), output_h_(output_h), output_w_(output_w) {
}

InferStatus AdaptAvgPooling::Forward(const std::vector<sftensor> &inputs,
                                     std::vector<sftensor> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array is empty";
    return InferStatus::InferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  CHECK(this->output_h_ > 0 && this->output_w_ > 0) << "Output shape error";

  const uint32_t batch = inputs.size();

#pragma omp parallel for num_threads(batch)
  for (uint32_t b = 0; b < batch; ++b) {
    const sftensor &input = inputs.at(b);
    CHECK(input != nullptr && !input->empty())
        << "The " << b << " input tensor is empty";

    const uint32_t input_c = input->channels();
    const uint32_t input_h = input->rows();
    const uint32_t input_w = input->cols();

    const uint32_t stride_h = uint32_t(std::floor(input_h / output_h_));
    const uint32_t stride_w = uint32_t(std::floor(input_w / output_w_));
    CHECK(stride_w > 0 && stride_h > 0)
        << "The stride of the " << b << " input tensor is wrong";

    const uint32_t kernel_h = input_h - (output_h_ - 1) * stride_h;
    const uint32_t kernel_w = input_w - (output_w_ - 1) * stride_w;
    CHECK(kernel_h > 0 && kernel_w > 0)
        << "The kernel of the " << b << " input tensor is wrong";

    sftensor &output = outputs.at(b);
    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The " << b << " output tensor is empty";
      output = std::make_shared<ftensor>(input_c, output_h_, output_w_);
    }

    CHECK(output->channels() == input_c || output->rows() == output_h_ ||
          output->cols() == output_w_)
        << "The " << b << " output tensor dimension is wrong";

    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat &in_channel = input->slice(ic);
      arma::fmat &out_channel = output->slice(ic);
      // 在输入tensor上滑动
      for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
        for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
          // 计算池化窗口内元素值总和
          float val_sum = 0.f;
          for (uint32_t w = 0; w < kernel_w; ++w) {
            const float *in_col_ptr = in_channel.colptr(c + w) + r;
            for (uint32_t h = 0; h < kernel_h; ++h) {
              float val = *(in_col_ptr + h);
              val_sum += val;
            }
          }
          // 计算平均值=元素值总和 / 池化窗口大小
          float *out_col_ptr = out_channel.colptr(int(c / stride_w));
          *(out_col_ptr + int(r / stride_h)) =
              val_sum / float(kernel_h * kernel_w);
        }
      }
    }
  }

  return InferStatus::InferSuccess;
}

ParseParamAttrStatus AdaptAvgPooling::GetInstance(const srunop &op,
                                                  slayer &adapt_avgpooling) {
  if (op == nullptr) {
    LOG(ERROR) << "Operator is empty";
    return ParseParamAttrStatus::OpEmpty;
  }

  const auto &params = op->params;

  if (params.find("output_size") == params.end()) {
    LOG(ERROR) << "Output size parameter is missing";
    return ParseParamAttrStatus::ParamMissingOutHW;
  }

  const auto &out_hw_param =
      dynamic_cast<RuntimeParamIntArr *>(params.at("output_size"));
  if (out_hw_param == nullptr) {
    LOG(ERROR) << "Output size parameter is missing";
    return ParseParamAttrStatus::ParamMissingOutHW;
  }

  const auto &out_hw = out_hw_param->value;
  if (out_hw.size() != 2) {
    LOG(ERROR) << "Output size parameter is missing";
    return ParseParamAttrStatus::ParamMissingOutHW;
  }

  adapt_avgpooling =
      std::make_shared<AdaptAvgPooling>(out_hw.at(0), out_hw.at(1));

  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

LayerRegisterWrapper AdaptAvgPoolingGetInstance("nn.AdaptiveAvgPool2d",
                                                AdaptAvgPooling::GetInstance);

} // namespace TinyInfer