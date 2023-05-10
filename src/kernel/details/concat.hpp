#ifndef TINY_INFER_SOURCE_KERNEL_DETAILS_CONCAT_HPP_
#define TINY_INFER_SOURCE_KERNEL_DETAILS_CONCAT_HPP_

#include "kernel/abstract/no_attr_kernel.hpp"

namespace TinyInfer {

class Concat : public NoAttrKernel {
public:
  explicit Concat(int dim = 0);

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus Creator(const srunop &op, skernel &concat);

private:
  int32_t dim_; // 拼接维度
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_DETAILS_CONCAT_HPP_
