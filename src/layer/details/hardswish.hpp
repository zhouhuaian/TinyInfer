#ifndef TINY_INFER_SOURCE_LAYER_HARDSWISH_HPP_
#define TINY_INFER_SOURCE_LAYER_HARDSWISH_HPP_

#include "layer/abstract/no_attr_layer.hpp"

namespace TinyInfer {

class HardSwish : public NoAttrLayer {
public:
  explicit HardSwish();

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus GetInstance(const srunop &op, slayer &hardswish);
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_LAYER_HARDSWISH_HPP_