#ifndef TINY_INFER_SOURCE_KERNEL_ADAPTAVGPOOLING_HPP_
#define TINY_INFER_SOURCE_KERNEL_ADAPTAVGPOOLING_HPP_

#include "kernel/abstract/no_attr_kernel.hpp"

namespace TinyInfer {

class AdaptAvgPooling : public NoAttrKernel {
public:
  explicit AdaptAvgPooling(uint32_t output_h = 0, uint32_t output_w = 0);

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus Creator(const srunop &op,
                                      skernel &adapt_avgpooling);

private:
  uint32_t output_h_; // 输出Tensor高度
  uint32_t output_w_; // 输出Tensor宽度
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_ADAPTAVGPOOLING_HPP_
