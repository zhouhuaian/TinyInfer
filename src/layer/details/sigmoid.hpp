#ifndef TINY_INFER_SOURCE_LAYER_SIGMOID_HPP_
#define TINY_INFER_SOURCE_LAYER_SIGMOID_HPP_

#include "layer/abstract/no_attr_layer.hpp"

namespace TinyInfer {

class Sigmoid : public NoAttrLayer {
public:
  explicit Sigmoid();

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus GetInstance(const srunop &op, slayer &sigmoid);
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_LAYER_SIGMOID_HPP_
