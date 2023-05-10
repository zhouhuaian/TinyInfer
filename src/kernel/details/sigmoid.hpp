#ifndef TINY_INFER_SOURCE_KERNEL_SIGMOID_HPP_
#define TINY_INFER_SOURCE_KERNEL_SIGMOID_HPP_

#include "kernel/abstract/no_attr_kernel.hpp"

namespace TinyInfer {

class Sigmoid : public NoAttrKernel {
public:
  explicit Sigmoid();

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus Creator(const srunop &op, skernel &sigmoid);
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_SIGMOID_HPP_
