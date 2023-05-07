#include "expression.hpp"
#include "data/tensor.hpp"
#include "kernel/abstract/kernel_factory.hpp"
#include "status_code.hpp"
#include <cstdint>
#include <glog/logging.h>
#include <memory>
#include <stack>

namespace TinyInfer {

Expression::Expression(std::string statement)
    : NoAttrKernel("Expression"),
      parser_(std::make_unique<ExprParser>(std::move(statement))) {}

InferStatus Expression::Forward(const std::vector<sftensor> &inputs,
                                std::vector<sftensor> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array is empty";
    return InferStatus::InferFailedInputEmpty;
  }

  // ! 和Concat层一样，Expression层（二元操作）要求输入和输出Tensor批次不相等
  if (outputs.empty() || outputs.size() == inputs.size() ||
      inputs.size() % outputs.size() != 0) {
    LOG(ERROR) << "The input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  const auto &in_batch = inputs.size();
  const auto &batch = outputs.size();

  for (uint32_t ib = 0; ib < in_batch; ++ib) {
    const auto &input = inputs.at(ib);
    if (input == nullptr || input->empty()) {
      LOG(ERROR) << "The " << ib << " input tensor is empty";
      return InferStatus::InferFailedInputEmpty;
    }
  }

  uint32_t channels = inputs.at(0)->channels();
  uint32_t rows = inputs.at(0)->rows();
  uint32_t cols = inputs.at(0)->cols();

  for (uint32_t b = 0; b < batch; ++b) {
    auto &output = outputs.at(b);
    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The " << b << " output tensor is empty";
      output = std::make_shared<ftensor>(channels, rows, cols);
    }
    output->Fill(0.f);
  }

  // 词法分析
  CHECK(this->parser_ != nullptr) << "The parser is empty";
  this->parser_->Tokenize(false);
  const auto &tokens = this->parser_->Tokens();
  CHECK(!tokens.empty()) << "Tokenize failed";

  std::stack<std::vector<sftensor>> stk;               // 运算数栈
  const auto &token_nodes = this->parser_->Generate(); // 获取逆波兰式
  // 依据逆波兰式执行表达式
  for (const auto &token_node : token_nodes) {
    // 运算数入栈
    if (token_node->num >= 0) {
      // !
      // inputs按顺序保存了所有来源的输入Tensor：[tensor1,tensor2,...,tensor1,tensor2,...]
      // !
      // 而num表示不同输入来源的下标——即RuntimeOperator的in_oprands_seq中的下标
      std::vector<sftensor> in_node(batch); // 存放同一来源的输入Tensor
      uint32_t start_idx = token_node->num * batch;
      for (uint32_t b = 0; b < batch; ++b) {
        CHECK(b + start_idx < in_batch);
        in_node.at(b) = inputs.at(b + start_idx);
      }

      stk.push(in_node); // 压入栈中
    }
    // 遇到运算符，弹出两个来源的输入Tensor，执行运算
    else {
      const int32_t op = token_node->num;
      CHECK(stk.size() >= 2) << "The number of operand is less than two";
      const auto in_node2 = stk.top();
      stk.pop();
      const auto in_node1 = stk.top();
      stk.pop();

      std::vector<sftensor> out_node(batch);

#pragma omp parallel for num_threads(batch)
      for (uint32_t b = 0; b < batch; ++b) {
        // Tensor相加
        if (op == int(TokenType::TokenAdd)) {
          out_node.at(b) = ElemAdd(in_node1.at(b), in_node2.at(b));
        }
        // Tensor相乘
        else if (op == int(TokenType::TokenMul)) {
          out_node.at(b) = ElemMul(in_node1.at(b), in_node2.at(b));
        } else {
          LOG(FATAL) << "Unsupported operation type: " << op;
        }
      }

      stk.push(out_node); // 将运算结果再压入栈中
    }
  }

  // 取出最终计算结果保存到输出Tensor中
  CHECK(stk.size() == 1);
  const auto output_node = stk.top();
  stk.pop();

  for (int b = 0; b < batch; ++b) {
    auto &output = outputs.at(b);
    CHECK(output->shape() == output_node.at(b)->shape());
    output = output_node.at(b);
  }

  return InferStatus::InferSuccess;
}

ParseParamAttrStatus Expression::GetInstance(const srunop &op,
                                             skernel &expression) {

  if (op == nullptr) {
    LOG(ERROR) << "Operator is empty";
    return ParseParamAttrStatus::OpEmpty;
  }

  const auto &params = op->params;

  if (params.find("expr") == params.end()) {
    LOG(ERROR) << "Expr parameter is missing";
    return ParseParamAttrStatus::ParamMissingExpr;
  }

  const auto &state_param = dynamic_cast<RuntimeParamStr *>(params.at("expr"));
  if (state_param == nullptr ||
      state_param->type != RuntimeParamType::ParamStr) {
    LOG(ERROR) << "Expr parameter is missing";
    return ParseParamAttrStatus::ParamMissingExpr;
  }

  expression = std::make_shared<Expression>(state_param->value);

  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

KernelRegisterWrapper ExpressionGetInstance("pnnx.Expression",
                                            Expression::GetInstance);

} // namespace TinyInfer
