#ifndef TINY_INFER_SOURCE_KERNEL_EXPRESSION_HPP_
#define TINY_INFER_SOURCE_KERNEL_EXPRESSION_HPP_

#include "kernel/abstract/no_attr_kernel.hpp"
#include "parser/parse_expr.hpp"

namespace TinyInfer {

class Expression : public NoAttrKernel {
public:
  explicit Expression(std::string statement);

  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  static ParseParamAttrStatus Creator(const srunop &op, skernel &expression);

private:
  std::unique_ptr<ExprParser> parser_; // 表达式解析器
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_EXPRESSION_HPP_
