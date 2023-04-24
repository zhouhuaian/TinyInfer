#include "flatten.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "status_code.hpp"
#include <glog/logging.h>

namespace TinyInfer {

Flatten::Flatten(int start_dim, int end_dim)
    : NoAttrLayer("Flatten"), start_dim_(start_dim), end_dim_(end_dim) {}

InferStatus Flatten::Forward(const std::vector<sftensor>& inputs, std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array is empty";
    return InferStatus::InferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  int total_dims = 4;  // (batchs, channels, rows, cols)
  int start_dim = start_dim_;
  int end_dim = end_dim_;

  // 将负的维度值转换为正数
  if (start_dim < 0) {
    start_dim = total_dims + start_dim;
  }
  if (end_dim < 0) {
    end_dim = total_dims + end_dim;
  }

  // ! 将维度值转换为单个Tensor内的维度（3维），而不是一个批次的维度（4维）
  start_dim -= 1;
  end_dim -= 1;
  
  CHECK(end_dim <= 2 && start_dim >= 0 && end_dim > start_dim)
      << "Flatten dimension error: " << "start_dim: " << start_dim_
      << " end_dim: " << end_dim_;

  const uint32_t batch = inputs.size();

#pragma omp parallel for num_threads(batch)
  for (uint32_t b = 0; b < batch; ++b) {
    const sftensor& input = inputs.at(b);
    CHECK(input != nullptr && !input->empty()) 
        << "The " << b << "th/st/nd input tensor is empty";

    // 统计flatten的元素数目
    const auto& in_shape = input->shape();
    uint32_t elem_ct = 1;
    for (int i = start_dim; i <= end_dim; ++i) {
      elem_ct *= in_shape.at(i);
    }

    // 将输入Tensor的元素拷贝给输出Tensor
    sftensor& output = outputs.at(b);
    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The " << b << "th/st/nd output tensor is empty";
      output = input->Clone();
    } 
    else {
      CHECK(input->size() == output->size()) 
          << "The " << b << "th/st/nd input and output tensor element size do not match";

      memcpy(output->data().memptr(), input->data().memptr(), sizeof(float) * input->size());
    }

    // 执行flatten操作
    // (channels, rows, cols)->(1, channels * rows * cols, 1)
    if (start_dim == 0 && end_dim == 2) {
      output->Reshape({elem_ct}, true);
    } 
    // (channels, rows, cols)->(1, channels, rows * cols)
    else if (start_dim == 1 && end_dim == 2) {
      uint32_t channels = input->channels();
      output->Reshape({channels, elem_ct}, true);
    } 
    // (channels, rows, cols)->(1, channels * rows, cols)
    else if (start_dim == 0 && end_dim == 1) {
      uint32_t cols = input->cols();
      output->Reshape({elem_ct, cols}, true);
    } 
  }
  
  return InferStatus::InferSuccess;
}

ParseParamAttrStatus Flatten::GetInstance(const srunop& op, slayer& flatten) {
  if (op == nullptr) {
    LOG(ERROR) << "Operator is empty";
    return ParseParamAttrStatus::OpEmpty;
  }  
  
  const auto& params = op->params;

  if (params.find("start_dim") == params.end()) {
    LOG(ERROR) << "Start dim parameter is missing";
    return ParseParamAttrStatus::ParamMissingDim;
  }

  const auto& start_dim =
      dynamic_cast<RuntimeParamInt*>(params.at("start_dim"));
  if (start_dim == nullptr) {
    LOG(ERROR) << "Start dim parameter is missing";
    return ParseParamAttrStatus::ParamMissingDim;
  }

  if (params.find("end_dim") == params.end()) {
    LOG(ERROR) << "End dim parameter is missing";
    return ParseParamAttrStatus::ParamMissingDim;
  }

  const auto& end_dim =
      dynamic_cast<RuntimeParamInt*>(params.at("end_dim"));
  if (end_dim == nullptr) {
    LOG(ERROR) << "End dim parameter is missing";
    return ParseParamAttrStatus::ParamMissingDim;
  }

  flatten = std::make_shared<Flatten>(start_dim->value, end_dim->value);
  
  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

LayerRegisterWrapper FlattenGetInstance("torch.flatten", Flatten::GetInstance);

}  // namespace TinyInfer