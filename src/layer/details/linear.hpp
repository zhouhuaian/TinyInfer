#ifndef TINY_INFER_SOURCE_LAYER_LINEAR_HPP_
#define TINY_INFER_SOURCE_LAYER_LINEAR_HPP_

#include "layer/abstract/attr_layer.hpp"

namespace TinyInfer {

class Linear : public AttrLayer {
public:

  explicit Linear(uint32_t in_features = 0, uint32_t out_features = 0, bool use_bias = false);

  InferStatus Forward(const std::vector<sftensor>& inputs, std::vector<sftensor>& outputs) override;

  static ParseParamAttrStatus GetInstance(const srunop& op, slayer& linear);
 
private:
  uint32_t in_features_;    // 输入特征长度
  uint32_t out_features_;   // 输出特征长度
  bool use_bias_;           // 含有偏置与否
};

}  // namespace TinyInfer

#endif  // TINY_INFER_SOURCE_LAYER_LINEAR_HPP_
