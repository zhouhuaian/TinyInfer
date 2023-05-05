#ifndef TINY_INFER_SOURCE_LAYER_HARDSIGMOID_HPP_
#define TINY_INFER_SOURCE_LAYER_HARDSIGMOID_HPP_

#include "layer/abstract/no_attr_layer.hpp"

namespace TinyInfer {

class HardSigmoid : public NoAttrLayer {
public:
  explicit HardSigmoid();

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus GetInstance(const srunop &op,
                                          slayer &hardsigmoid);
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_LAYER_HARDSIGMOID_HPP_
