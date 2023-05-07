#include "concat.hpp"
#include "kernel/abstract/kernel_factory.hpp"
#include "status_code.hpp"

namespace TinyInfer {

Concat::Concat(int dim) : NoAttrKernel("Concat"), dim_(dim) {}

InferStatus Concat::Forward(const std::vector<sftensor> &inputs,
                            std::vector<sftensor> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array is empty";
    return InferStatus::InferFailedInputEmpty;
  }

  // ! Concat层要求输入Tensor和输出Tensor批次不相等
  if (outputs.size() == inputs.size() || outputs.empty() ||
      inputs.size() % outputs.size() != 0) {
    LOG(ERROR) << "The input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  // ! Concat目前只支持沿着Channel维度将Tensor拼接
  CHECK(dim_ == 1 || dim_ == -3) << "Concat dimension error";

  // !
  // 注意inputs保存了多个来源的输入Tensor，而Concat就是把多个来源的输入Tensor按Channel维度拼接
  const uint32_t in_batch = inputs.size();
  const uint32_t out_batch = outputs.size();
  const uint32_t packet_sz = in_batch / out_batch; // packet_sz是输入来源数目

  uint32_t rows = inputs.front()->rows();
  uint32_t cols = inputs.front()->cols();

#pragma omp parallel for num_threads(out_batch)
  for (uint32_t b = 0; b < out_batch; ++b) {
    sftensor &output = outputs.at(b);
    uint32_t idx_c = 0; // 用于记录输出Tensor已拼接的Channel数目

    for (uint32_t ib = b; ib < in_batch; ib += out_batch) {
      const sftensor &input = inputs.at(ib);
      CHECK(input != nullptr && !input->empty())
          << "The " << ib << " input tensor is empty";

      CHECK(input->rows() == rows && input->cols() == cols)
          << "The " << ib << " input tensor dimension is wrong";

      const uint32_t in_channels = input->channels();

      if (output == nullptr || output->empty()) {
        DLOG(ERROR) << "The " << b << " output tensor is empty";
        output = std::make_shared<ftensor>(in_channels * packet_sz, rows, cols);
      }

      CHECK(output->channels() == in_channels * packet_sz &&
            output->rows() == rows && output->cols() == cols)
          << "The " << b << " output tensor dimension is wrong";

      for (uint32_t ic = 0; ic < in_channels; ++ic) {
        output->slice(idx_c + ic) = input->slice(ic);
      }
      idx_c += in_channels;
    }
  }

  return InferStatus::InferSuccess;
}

ParseParamAttrStatus Concat::GetInstance(const srunop &op, skernel &concat) {
  if (op == nullptr) {
    LOG(ERROR) << "Operator is nullptr";
    return ParseParamAttrStatus::OpEmpty;
  }

  const auto &params = op->params;

  if (params.find("dim") == params.end()) {
    LOG(ERROR) << "Dim parameter is missing";
    return ParseParamAttrStatus::ParamMissingDim;
  }

  const auto &dim_param = dynamic_cast<RuntimeParamInt *>(params.at("dim"));
  if (dim_param == nullptr) {
    LOG(ERROR) << "Dim parameter is missing";
    return ParseParamAttrStatus::ParamMissingDim;
  }

  concat = std::make_shared<Concat>(dim_param->value);

  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

KernelRegisterWrapper ConcatGetInstance("torch.cat", Concat::GetInstance);

} // namespace TinyInfer