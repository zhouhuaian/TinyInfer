#ifndef TINY_INFER_INCLUDE_PARSER_PARSE_EXPR_HPP_
#define TINY_INFER_INCLUDE_PARSER_PARSE_EXPR_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace TinyInfer {

// Token类型
enum class TokenType {
  TokenUnknown = -1,
  TokenInputNum = -2, // 运算数
  TokenComma = -3,
  TokenLBracket = -4, // 左括号
  TokenRBracket = -5, // 右括号
  TokenAdd = -6,
  TokenMul = -7,
};

// Token类
struct Token {
  Token(TokenType type = TokenType::TokenUnknown, int32_t start = 0,
        int32_t end = 0)
      : type(type), start(start), end(end) {}

  TokenType type; // Token类型
  int32_t start;  // Token开始位置
  int32_t end;    // Token结束位置 [start,end)
};

// AST节点
struct TokenNode {
  TokenNode(int32_t num = -1, std::shared_ptr<TokenNode> left = nullptr,
            std::shared_ptr<TokenNode> right = nullptr)
      : num(num), left(left), right(right) {}

  int32_t num; // 节点值——正数表示运算数，负数表示运算符
  std::shared_ptr<TokenNode> left;  // 左子节点
  std::shared_ptr<TokenNode> right; // 右子节点
};

using stokennode = std::shared_ptr<TokenNode>;

// 表达式解析器类
class ExprParser {
public:
  explicit ExprParser(std::string statement)
      : statement(std::move(statement)) {}

  /**
   * 词法分析——分词
   * @param retokenize 是否重新进行词法分析
   */
  void Tokenize(bool retokenize = false);

  /**
   * 语法分析——生成表达式的逆波兰式
   * @return 逆波兰式
   */
  std::vector<stokennode> Generate();

  /**
   * 返回Tokens
   */
  const std::vector<Token> &Tokens() const;

  /**
   * 返回Tokens对应的字符串
   */
  const std::vector<std::string> &TokenStrs() const;

private:
  /**
   * 生成AST
   * @param index token在tokens中的下标
   * @return 生成的AST
   */
  stokennode Generate_(int32_t &index);

  std::string statement;               // 待解析的表达式
  std::vector<Token> tokens;           // 词法分析得到的Tokens
  std::vector<std::string> token_strs; // Tokens对应的字符串
};

/**
 * 由AST生成表达式的逆波兰式
 * @param root AST根节点
 * @param re_polish 逆波兰式
 */
void ReversePolish(const stokennode &root, std::vector<stokennode> &re_polish);

} // namespace TinyInfer

#endif // TINY_INFER_INCLUDE_PARSER_PARSE_EXPR_HPP_
