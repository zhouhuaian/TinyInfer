#include "layer/abstract/layer.hpp"

namespace TinyInfer {

InferStatus Layer::Forward() {
  CHECK(!this->op_.expired()) << "Runtime operator is expired or nullptr";

  // 获取计算节点
  const auto &op = this->op_.lock();
  // 获取Layer的所有输入操作数
  const std::vector<srunoprand> &in_oprands = op->in_oprands_seq;

  // ! 准备Layer的输入Tensor——将所有来源的输入Tensor按顺序保存在同一个vector中
  std::vector<sftensor> in_datas;
  for (const auto &in_oprand : in_oprands) {
    for (const auto &input_data : in_oprand->data) {
      in_datas.push_back(input_data);
    }
  }

  // 分别检查输入输出空间是否为空
  CHECK(!in_datas.empty()) << op->name << " operator input data is empty";
  CHECK(op->out_oprand != nullptr && !op->out_oprand->data.empty())
      << op->name << " operator output data is empty";

  // Layer执行推理
  InferStatus status = this->Forward(in_datas, op->out_oprand->data);
  return status;
}

InferStatus Layer::Forward(const std::vector<sftensor> &inputs,
                           std::vector<sftensor> &outputs) {
  LOG(FATAL) << this->name_ << " layer not implement yet!";
}

void Layer::set_weights(const std::vector<sftensor> &weights) {
  LOG(FATAL) << this->name_ << " layer not implement yet!";
}

void Layer::set_weights(const std::vector<float> &weights) {
  LOG(FATAL) << this->name_ << " layer not implement yet!";
}

const std::vector<sftensor> &Layer::weights() const {
  LOG(FATAL) << this->name_ << " layer not implement yet!";
}

void Layer::set_bias(const std::vector<sftensor> &bias) {
  LOG(FATAL) << this->name_ << " layer not implement yet!";
}

void Layer::set_bias(const std::vector<float> &bias) {
  LOG(FATAL) << this->name_ << " layer not implement yet!";
}

const std::vector<sftensor> &Layer::bias() const {
  LOG(FATAL) << this->name_ << " layer not implement yet!";
}

void Layer::set_runtime_op(const srunop &op) { this->op_ = op; }

} // namespace TinyInfer