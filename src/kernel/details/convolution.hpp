#ifndef TINY_INFER_SOURCE_KERNEL_CONVOLUTION_HPP_
#define TINY_INFER_SOURCE_KERNEL_CONVOLUTION_HPP_

#include "kernel/abstract/attr_kernel.hpp"
#include <cstdint>

namespace TinyInfer {

class Convolution : public AttrKernel {
public:
  explicit Convolution(uint32_t out_channels, uint32_t in_channels,
                       uint32_t kernel_h, uint32_t kernel_w,
                       uint32_t padding_h = 0, uint32_t padding_w = 0,
                       uint32_t stride_h = 1, uint32_t stride_w = 1,
                       uint32_t groups = 1, bool use_bias = false);

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus GetInstance(const srunop &op,
                                          skernel &convolution);

private:
  uint32_t padding_h_;
  uint32_t padding_w_;
  uint32_t stride_h_;
  uint32_t stride_w_;
  uint32_t groups_; // 分组卷积的组数
  bool use_bias_;
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_CONVOLUTION_HPP_
