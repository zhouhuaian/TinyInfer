#ifndef TINY_INFER_SOURCE_LAYER_MAXPOOLING_HPP_
#define TINY_INFER_SOURCE_LAYER_MAXPOOLING_HPP_

#include "layer/abstract/no_attr_layer.hpp"

namespace TinyInfer {

class MaxPooling : public NoAttrLayer {
public:
  explicit MaxPooling(uint32_t padding_h = 0, uint32_t padding_w = 0,
                      uint32_t kernel_h = 0, uint32_t kernel_w = 0,
                      uint32_t stride_h = 1, uint32_t stride_w = 1);

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus GetInstance(const srunop &op, slayer &maxpooling);

private:
  uint32_t padding_h_; // 高度填充值
  uint32_t padding_w_; // 宽度填充值
  uint32_t kernel_h_;  // 池化核高度
  uint32_t kernel_w_;  // 池化核宽度
  uint32_t stride_h_;  // 高度方向步长
  uint32_t stride_w_;  // 宽度方向步长
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_LAYER_MAXPOOLING_HPP_
