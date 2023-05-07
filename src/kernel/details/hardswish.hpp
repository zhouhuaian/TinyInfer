#ifndef TINY_INFER_SOURCE_KERNEL_HARDSWISH_HPP_
#define TINY_INFER_SOURCE_KERNEL_HARDSWISH_HPP_

#include "kernel/abstract/no_attr_kernel.hpp"

namespace TinyInfer {

class HardSwish : public NoAttrKernel {
public:
  explicit HardSwish();

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus GetInstance(const srunop &op, skernel &hardswish);
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_HARDSWISH_HPP_