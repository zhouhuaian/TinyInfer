#ifndef TINY_INFER_SOURCE_KERNEL_HARDSIGMOID_HPP_
#define TINY_INFER_SOURCE_KERNEL_HARDSIGMOID_HPP_

#include "kernel/abstract/no_attr_kernel.hpp"

namespace TinyInfer {

class HardSigmoid : public NoAttrKernel {
public:
  explicit HardSigmoid();

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus Creator(const srunop &op, skernel &hardsigmoid);
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_HARDSIGMOID_HPP_
