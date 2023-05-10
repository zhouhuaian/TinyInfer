#ifndef TINY_INFER_SOURCE_KERNEL_SOFTMAX_HPP_
#define TINY_INFER_SOURCE_KERNEL_SOFTMAX_HPP_

#include "kernel/abstract/no_attr_kernel.hpp"

namespace TinyInfer {

class Softmax : public NoAttrKernel {
public:
  explicit Softmax(int dim = -1);

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus Creator(const srunop &op, skernel &softmax);

private:
  int dim_; // 执行softmax操作的维度，-1表示倒数第一个维度
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_SOFTMAX_HPP_
