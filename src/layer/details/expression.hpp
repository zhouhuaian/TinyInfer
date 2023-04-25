#ifndef TINY_INFER_SOURCE_LAYER_EXPRESSION_HPP_
#define TINY_INFER_SOURCE_LAYER_EXPRESSION_HPP_

#include "layer/abstract/no_attr_layer.hpp"
#include "parser/parse_expr.hpp"

namespace TinyInfer {

class Expression : public NoAttrLayer {
public:
  explicit Expression(std::string statement);

  InferStatus Forward(const std::vector<sftensor>& inputs, std::vector<sftensor>& outputs) override;

  static ParseParamAttrStatus GetInstance(const srunop& op, slayer& expression);

private:
  std::unique_ptr<ExprParser> parser_;  // 表达式解析器
};

}  // namespace TinyInfer

#endif  // TINY_INFER_SOURCE_LAYER_EXPRESSION_HPP_
