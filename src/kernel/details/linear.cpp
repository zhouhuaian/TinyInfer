#include "linear.hpp"
#include "data/tensor.hpp"
#include "kernel/abstract/kernel_factory.hpp"
#include "runtime/runtime_attr.hpp"
#include "status_code.hpp"
#include <cstddef>
#include <cstdint>
#include <glog/logging.h>

namespace TinyInfer {

Linear::Linear(uint32_t in_features, uint32_t out_features, bool use_bias)
    : AttrKernel("Linear"), in_features_(in_features),
      out_features_(out_features), use_bias_(use_bias) {

  // Linear层仅有一个权重Tensor和最多一个偏置Tensor
  this->InitWeights(1, 1, out_features_, in_features_);

  if (use_bias) {
    this->InitBias(1, 1, out_features_, 1);
  }
}

InferStatus Linear::Forward(const std::vector<sftensor> &inputs,
                            std::vector<sftensor> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array is empty";
    return InferStatus::InferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  CHECK(!this->weights_.empty() && this->weights_.size() == 1)
      << "The weight count must be 1";

  const auto &weight_tensor = weights_.front();
  CHECK(!weight_tensor->empty()) << "The weight is empty";

  arma::fmat weight(weight_tensor->data().memptr(), out_features_, in_features_,
                    false, true);
  CHECK(weight.n_rows == out_features_ && weight.n_cols == in_features_)
      << "Weight shape error";

  if (this->use_bias_) {
    CHECK(this->bias_.size() == this->weights_.size())
        << "The bias count is not 1";
  }

  uint32_t batch = inputs.size();
#pragma omp parallel for num_threads(batch)
  for (uint32_t b = 0; b < batch; ++b) {
    const sftensor &input = inputs.at(b);
    sftensor &output = outputs.at(b);

    CHECK(input != nullptr && !input->empty())
        << "The " << b << " input tensor is empty";

    const std::vector<uint32_t> &in_shape = input->shape();
    // 检查输入Tensor的通道数和特征长度
    CHECK(in_shape.size() == 3 && in_shape.at(0) == 1 &&
          in_shape.at(1) == in_features_)
        << "The " << b << " input tensor dimension is wrong";

    // ! 输入特征的数目（一个输入特征为一个列向量），注意：可能有多个输入特征
    const uint32_t in_dims = in_shape.at(2);

    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The " << b << " output tensor is empty";
      output = std::make_shared<ftensor>(1, out_features_, in_dims);
    }

    CHECK(output->channels() == 1 && output->rows() == out_features_ &&
          output->cols() == in_dims)
        << "The " << b << " output tensor dimension is wrong";

    // 执行矩阵计算
    arma::fmat col_vec(input->data().memptr(), in_features_, in_dims, false,
                       true);
    arma::fmat &result = output->slice(0);
    result = weight * col_vec;

    // 加上偏置
    if (use_bias_) {
      const auto &bias_tensor = this->bias_.front();
      CHECK(!bias_tensor->empty()) << "The bias is empty";

      const auto &bias = bias_tensor->data();
      CHECK(bias.n_slices == 1 && bias.n_rows == out_features_);

      result += bias.slice(0);
    }
  }

  return InferStatus::InferSuccess;
}

ParseParamAttrStatus Linear::Creator(const srunop &op, skernel &linear) {
  if (op == nullptr) {
    LOG(ERROR) << "Operator is empty";
    return ParseParamAttrStatus::OpEmpty;
  }

  const auto &params = op->params;

  if (params.find("bias") == params.end()) {
    LOG(ERROR) << "Bias parameter is missing";
    return ParseParamAttrStatus::ParamMissingBias;
  }

  const auto &bias_param = dynamic_cast<RuntimeParamBool *>(params.at("bias"));
  if (bias_param == nullptr) {
    LOG(ERROR) << "Bias parameter is missing";
    return ParseParamAttrStatus::ParamMissingBias;
  }

  const auto &attr = op->attrs;
  // 取出权重和偏置
  if (attr.find("weight") == attr.end()) {
    LOG(ERROR) << "Weight attribute is missing";
    return ParseParamAttrStatus::AttrMissingWeight;
  }

  const auto &weight = attr.at("weight");
  if (weight == nullptr) {
    LOG(ERROR) << "Weight attribute is missing";
    return ParseParamAttrStatus::AttrMissingWeight;
  }

  srunattr bias;
  if (bias_param->value) {
    if (attr.find("bias") == attr.end()) {
      LOG(ERROR) << "Bias attribute is missing";
      return ParseParamAttrStatus::AttrMissingBias;
    }

    bias = attr.at("bias");
    if (bias == nullptr) {
      LOG(ERROR) << "Bias attribute is missing";
      return ParseParamAttrStatus::AttrMissingBias;
    }
  }

  const auto &shape = weight->shape;
  if (shape.size() < 2) {
    LOG(ERROR) << "Output features size is missing";
    return ParseParamAttrStatus::AttrMissingOutFeatures;
  }

  int32_t out_features = shape.at(0);
  int32_t in_features = shape.at(1);
  const bool use_bias = bias_param->value;

  linear = std::make_shared<Linear>(in_features, out_features, use_bias);

  // 加载权重、偏置
  linear->set_weights(weight->get<float>());

  if (use_bias) {
    linear->set_bias(bias->get<float>());
  }

  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

KernelRegisterWrapper LinearCreator("nn.Linear", Linear::Creator);

} // namespace TinyInfer