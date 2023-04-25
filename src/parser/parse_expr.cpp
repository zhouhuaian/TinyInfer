#include <algorithm>
#include <cctype>
#include <stack>
#include <utility>
#include <glog/logging.h>
#include "parser/parse_expr.hpp"

namespace TinyInfer {

void ExprParser::Tokenize(bool retokenize) {
  // 分词已完成，直接返回
  if (retokenize == false && !this->tokens.empty()) {
    return;
  }

  // 检查表达式是否为空
  CHECK(!statement.empty()) << "The input statement is empty!";
  
  // 移除表达式中的空格
  // ! remove_if删除元素时是将pred为假的元素移动到序列前部，
  // ! 所以需要erase删除后部不再使用的空间—即erase-remove idiom
  statement.erase(std::remove_if(statement.begin(), statement.end(), 
          [] (char c) { return std::isspace(c); }), statement.end());
  CHECK(!statement.empty()) << "The input statement is empty!";

  // 词法分析
  for (int32_t i = 0; i < statement.size(); ) {
    char c = statement.at(i);
    // 'add'
    if (c == 'a') {
      CHECK(i + 1 < statement.size() && statement.at(i + 1) == 'd')
              << "Parse add token failed, illegal character: " << statement.at(i + 1);
      CHECK(i + 2 < statement.size() && statement.at(i + 2) == 'd')
              << "Parse add token failed, illegal character: " << statement.at(i + 2);
      
      // 保存add token及字符串
      Token token(TokenType::TokenAdd, i, i + 3);
      tokens.push_back(token);
      token_strs.push_back(statement.substr(i, 3));
      i = i + 3;
    }
    // 'mul' 
    else if (c == 'm') {
      CHECK(i + 1 < statement.size() && statement.at(i + 1) == 'u')
              << "Parse mul token failed, illegal character: " << statement.at(i + 1);
      CHECK(i + 2 < statement.size() && statement.at(i + 2) == 'l')
              << "Parse mul token failed, illegal character: " << statement.at(i + 2);
      
      Token token(TokenType::TokenMul, i, i + 3);
      tokens.push_back(token);
      token_strs.push_back(statement.substr(i, 3));
      i = i + 3;
    }
    // '@123'
    else if (c == '@') {
      CHECK(i + 1 < statement.size() && std::isdigit(statement.at(i + 1)))
              << "Parse number token failed, illegal character: " << statement.at(i + 1);
      // 保存运算数token
      int32_t j = i + 1;
      for ( ; j < statement.size() && std::isdigit(statement.at(j)); ++j) {}
      
      Token token(TokenType::TokenInputNum, i, j);
      tokens.push_back(token);
      token_strs.push_back(statement.substr(i, j - i));
      i = j;
    }
    // ','
    else if (c == ',') {
      Token token(TokenType::TokenComma, i, i + 1);
      tokens.push_back(token);
      token_strs.push_back(statement.substr(i, 1));
      i += 1;
    }
    // '(' 
    else if (c == '(') {
      Token token(TokenType::TokenLBracket, i, i + 1);
      tokens.push_back(token);
      token_strs.push_back(statement.substr(i, 1));
      i += 1;
    }
    // ')' 
    else if (c == ')') {
      Token token(TokenType::TokenRBracket, i, i + 1);
      tokens.push_back(token);
      token_strs.push_back(statement.substr(i, 1));
      i += 1;
    } 
    else {
      LOG(FATAL) << "Unsupported illegal character: " << c;
    }
  }
}

std::vector<stokennode> ExprParser::Generate() {
  if (this->tokens.empty()) {
    this->Tokenize(true);
  }

  int index = 0;  // token在tokens中的下标
  stokennode root = Generate_(index);  // 生成AST
  CHECK(root != nullptr && index == tokens.size() - 1) 
      << "AST construct failed!";

  std::vector<stokennode> re_polish;
  ReversePolish(root, re_polish);  // 生成逆波兰式

  return re_polish;
}

const std::vector<Token>& ExprParser::Tokens() const { return this->tokens; }

const std::vector<std::string>& ExprParser::TokenStrs() const { return this->token_strs; }

stokennode ExprParser::Generate_(int32_t& index) {
  CHECK(index < this->tokens.size()) << "Token index error!";
  
  // 取出token，检查token类型是否为运算数或运算符
  const auto& token = this->tokens.at(index);
  const auto& type = token.type;
  CHECK(type == TokenType::TokenInputNum || type == TokenType::TokenAdd || 
        type == TokenType::TokenMul) << "Token type error!";
  
  // 由运算数token生成叶子节点
  if (type == TokenType::TokenInputNum) {
    uint32_t start = token.start + 1;  // 跳过'@'
    uint32_t end = token.end;
    CHECK(end > start && end <= this->statement.size()) 
        << "Number token size error!";
    
    const std::string& num_str = this->statement.substr(start, end - start);
    return std::make_shared<TokenNode>(std::stoi(num_str), nullptr, nullptr);
  } 
  // 由运算符token生成内部节点，并递归生成左右子树
  else if (type == TokenType::TokenMul || type == TokenType::TokenAdd) {
    stokennode node = std::make_shared<TokenNode>();
    node->num = int(type);

    index += 1;
    CHECK(index < this->tokens.size() && 
          this->tokens.at(index).type == TokenType::TokenLBracket) 
          << "Left bracket missing!";

    index += 1;
    CHECK(index < this->tokens.size()) << "Correspond left token missing!";
    
    const auto& left_token = this->tokens.at(index);  
    // 递归生成左子树
    if (left_token.type == TokenType::TokenInputNum || 
        left_token.type == TokenType::TokenAdd || 
        left_token.type == TokenType::TokenMul) {
      node->left = Generate_(index);  
    } else {
      LOG(FATAL) << "Unknown token type: " << int(left_token.type);
    }

    index += 1;
    CHECK(index < this->tokens.size() && 
          this->tokens.at(index).type == TokenType::TokenComma) 
          << "Comma missing!";

    index += 1;
    CHECK(index < this->tokens.size()) << "Correspond right token missing!";
    
    const auto& right_token = this->tokens.at(index);
    // 递归生成右子树
    if (right_token.type == TokenType::TokenInputNum || 
        right_token.type == TokenType::TokenAdd || 
        right_token.type == TokenType::TokenMul) {
      node->right = Generate_(index);
    } else {
      LOG(FATAL) << "Unknown token type: " << int(right_token.type);
    }

    index += 1;
    CHECK(index < this->tokens.size() && 
          this->tokens.at(index).type == TokenType::TokenRBracket) 
          << "Right bracket missing!";

    return node;
  } 
  else {
    LOG(FATAL) << "Unknown token type: " << int(type);
  }
}

void ReversePolish(const stokennode& root, std::vector<stokennode>& re_polish) {
  // 后序遍历AST获取逆波兰式
  if (root != nullptr) {
    ReversePolish(root->left, re_polish);
    ReversePolish(root->right, re_polish);
    re_polish.push_back(root);
  }
}

}  // namespace TinyInfer