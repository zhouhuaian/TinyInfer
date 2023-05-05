#ifndef TINY_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#define TINY_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_

#include "layer/abstract/no_attr_layer.hpp"

namespace TinyInfer {

class Flatten : public NoAttrLayer {
public:
  explicit Flatten(int start_dim = 0, int end_dim = 0);

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus GetInstance(const srunop &op, slayer &flatten);

private:
  int start_dim_; // 执行flatten的开始维度
  int end_dim_;   // 执行flatten的结束维度
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
