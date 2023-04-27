#ifndef TINY_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
#define TINY_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_

#include "layer/abstract/no_attr_layer.hpp"

namespace TinyInfer {

class Cat : public NoAttrLayer {
public:
  explicit Cat(int dim = 0);

  InferStatus Forward(const std::vector<sftensor>& inputs, std::vector<sftensor>& outputs) override;

  static ParseParamAttrStatus GetInstance(const srunop& op, slayer& cat);

private:
  int32_t dim_;  // 拼接维度
};

}  // namespace TinyInfer

#endif  // TINY_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
