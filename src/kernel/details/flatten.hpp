#ifndef TINY_INFER_SOURCE_KERNEL_DETAILS_FLATTEN_HPP_
#define TINY_INFER_SOURCE_KERNEL_DETAILS_FLATTEN_HPP_

#include "kernel/abstract/no_attr_kernel.hpp"

namespace TinyInfer {

class Flatten : public NoAttrKernel {
public:
  explicit Flatten(int start_dim = 0, int end_dim = 0);

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus GetInstance(const srunop &op, skernel &flatten);

private:
  int start_dim_; // 执行flatten的开始维度
  int end_dim_;   // 执行flatten的结束维度
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_DETAILS_FLATTEN_HPP_
