#include "hardswish.hpp"
#include "kernel/abstract/kernel_factory.hpp"
#include <glog/logging.h>

namespace TinyInfer {

HardSwish::HardSwish() : NoAttrKernel("HardSwish"){};

InferStatus HardSwish::Forward(const std::vector<sftensor> &inputs,
                               std::vector<sftensor> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array is empty";
    return InferStatus::InferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  const uint32_t batch = inputs.size();

#pragma omp parallel for num_threads(batch)
  for (uint32_t b = 0; b < batch; ++b) {
    const sftensor &input = inputs.at(b);
    sftensor &output = outputs.at(b);

    CHECK(input != nullptr && !input->empty())
        << "The " << b << " input tensor is empty";

    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The " << b << " output tensor is empty";
      output = std::make_shared<ftensor>(input->shape());
    }

    CHECK(input->shape() == output->shape())
        << "The " << b << " input and output tensor shape do not match";

    for (uint32_t i = 0; i < input->size(); ++i) {
      float in = input->index(i);
      float out = 0.f;
      if (in <= -3.f) {
        out = 0.f;
      } else if (in >= 3.f) {
        out = in;
      } else {
        out = in * (in + 3) / 6;
      }
      output->index(i) = out;
    }
  }

  return InferStatus::InferSuccess;
}

ParseParamAttrStatus HardSwish::Creator(const srunop &op, skernel &hardswish) {
  if (op == nullptr) {
    LOG(ERROR) << "Operator is empty";
    return ParseParamAttrStatus::OpEmpty;
  }

  hardswish = std::make_shared<HardSwish>();
  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

KernelRegisterWrapper HardSwishCreator("nn.Hardswish", HardSwish::Creator);

} // namespace TinyInfer