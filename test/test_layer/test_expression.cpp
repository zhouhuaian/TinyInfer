#include "../../src/layer/details/expression.hpp"
#include "data/load_data.hpp"
#include "parser/parse_expr.hpp"
#include "runtime/runtime_ir.hpp"
#include <gtest/gtest.h>

using namespace TinyInfer;

TEST(test_expression, add1) {
  RuntimeGraph graph("../../tmp/add/resnet_add.pnnx.param",
                     "../../tmp/add/resnet_add.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  const int batch = 4;
  std::vector<sftensor> inputs;
  inputs.reserve(batch);
  for (int b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(1, 4, 4);
    input->Fill(1.);
    inputs.push_back(input);
  }

  std::vector<sftensor> outputs = graph.Forward(inputs, false);
  ASSERT_EQ(outputs.size(), 4);

  const auto &output1 = outputs.at(0)->slice(0);
  const auto &output2 = CSVDataLoader::LoadData("../../tmp/add/1.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_LE(abs(output1.at(i) - output2.at(i)), 5e-6);
  }
}

TEST(test_expression, add2) {
  RuntimeGraph graph("../../tmp/add/resnet_add2.pnnx.param",
                     "../../tmp/add/resnet_add2.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  const int batch = 4;
  std::vector<sftensor> inputs;
  inputs.reserve(batch);
  for (int b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(1, 4, 4);
    input->Fill(1.);
    inputs.push_back(input);
  }

  std::vector<sftensor> outputs = graph.Forward(inputs, false);
  ASSERT_EQ(outputs.size(), 4);

  const auto &output1 = outputs.at(0)->slice(0);
  const auto &output2 = CSVDataLoader::LoadData("../../tmp/add/3.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_LE(abs(output1.at(i) - output2.at(i)), 5e-6);
  }
}

TEST(test_expression, mul1) {
  RuntimeGraph graph("../../tmp/add/resnet_add3.pnnx.param",
                     "../../tmp/add/resnet_add3.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  const int batch = 4;
  std::vector<sftensor> inputs;
  inputs.reserve(batch);
  for (int i = 0; i < batch; ++i) {
    sftensor input = std::make_shared<ftensor>(1, 4, 4);
    input->Fill(1.);
    inputs.push_back(input);
  }

  std::vector<sftensor> outputs = graph.Forward(inputs, false);
  ASSERT_EQ(outputs.size(), 4);

  const auto &output1 = outputs.at(0)->slice(0);
  const auto &output2 = CSVDataLoader::LoadData("../../tmp/add/7.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_LE(abs(output1.at(i) - output2.at(i)), 5e-6);
  }
}

TEST(test_layer, add1) {
  const std::string &str = "add(@0,@1)";
  Expression expression(str);
  sftensor input1 = std::make_shared<ftensor>(3, 224, 224);
  input1->Fill(1.f);
  sftensor input2 = std::make_shared<ftensor>(3, 224, 224);
  input2->Fill(1.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);

  std::vector<sftensor> outputs(1);
  outputs.at(0) = std::make_shared<ftensor>(3, 224, 224);
  const auto status = expression.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  sftensor output2 = std::make_shared<ftensor>(3, 224, 224);
  output2->Fill(2.f);

  sftensor output1 = outputs.front();
  ASSERT_TRUE(
      arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, add2) {
  const std::string &str = "add(@0,@1)";
  Expression expression(str);
  sftensor input1 = std::make_shared<ftensor>(3, 224, 224);
  input1->Fill(2.f);
  sftensor input2 = std::make_shared<ftensor>(3, 224, 224);
  input2->Fill(3.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);

  std::vector<sftensor> outputs(1);
  outputs.at(0) = std::make_shared<ftensor>(3, 224, 224);
  const auto status = expression.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  sftensor output2 = std::make_shared<ftensor>(3, 224, 224);
  output2->Fill(5.f);
  sftensor output1 = outputs.front();

  ASSERT_TRUE(
      arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, mul1) {
  const std::string &str = "mul(@0,@1)";
  Expression expression(str);
  sftensor input1 = std::make_shared<ftensor>(3, 224, 224);
  input1->Fill(2.f);
  sftensor input2 = std::make_shared<ftensor>(3, 224, 224);
  input2->Fill(3.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);

  std::vector<sftensor> outputs(1);
  outputs.at(0) = std::make_shared<ftensor>(3, 224, 224);
  const auto status = expression.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  sftensor output2 = std::make_shared<ftensor>(3, 224, 224);
  output2->Fill(6.f);
  sftensor output1 = outputs.front();

  ASSERT_TRUE(
      arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex1) {
  const std::string &str = "mul(@2,add(@0,@1))";
  Expression expression(str);
  sftensor input1 = std::make_shared<ftensor>(3, 224, 224);
  input1->Fill(2.f);
  sftensor input2 = std::make_shared<ftensor>(3, 224, 224);
  input2->Fill(3.f);

  sftensor input3 = std::make_shared<ftensor>(3, 224, 224);
  input3->Fill(4.f);

  std::vector<sftensor> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);

  std::vector<sftensor> outputs(1);
  outputs.at(0) = std::make_shared<ftensor>(3, 224, 224);
  const auto status = expression.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  sftensor output2 = std::make_shared<ftensor>(3, 224, 224);
  output2->Fill(20.f);
  sftensor output1 = outputs.front();

  ASSERT_TRUE(
      arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex2) {
  const std::string &str = "mul(add(@0,@1),add(@2,@3))";
  Expression expression(str);
  sftensor input1 = std::make_shared<ftensor>(3, 224, 224);
  input1->Fill(2.f);
  sftensor input2 = std::make_shared<ftensor>(3, 224, 224);
  input2->Fill(3.f);

  sftensor input3 = std::make_shared<ftensor>(3, 224, 224);
  input3->Fill(4.f);

  sftensor input4 = std::make_shared<ftensor>(3, 224, 224);
  input4->Fill(4.f);

  std::vector<sftensor> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

  std::vector<sftensor> outputs(1);
  outputs.at(0) = std::make_shared<ftensor>(3, 224, 224);
  const auto status = expression.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  sftensor output2 = std::make_shared<ftensor>(3, 224, 224);
  output2->Fill(40.f);
  sftensor output1 = outputs.front();

  ASSERT_TRUE(
      arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex3) {
  const std::string &str = "mul(mul(@0,@1),add(@2,@3))";
  Expression expression(str);
  sftensor input1 = std::make_shared<ftensor>(3, 224, 224);
  input1->Fill(2.f);
  sftensor input2 = std::make_shared<ftensor>(3, 224, 224);
  input2->Fill(3.f);

  sftensor input3 = std::make_shared<ftensor>(3, 224, 224);
  input3->Fill(4.f);

  sftensor input4 = std::make_shared<ftensor>(3, 224, 224);
  input4->Fill(4.f);

  std::vector<sftensor> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

  std::vector<sftensor> outputs(1);
  outputs.at(0) = std::make_shared<ftensor>(3, 224, 224);
  const auto status = expression.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  sftensor output2 = std::make_shared<ftensor>(3, 224, 224);
  output2->Fill(48.f);
  sftensor output1 = outputs.front();

  ASSERT_TRUE(
      arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex4) {
  const std::string &str = "mul(mul(@0,@1), mul(@2,@3))";
  Expression expression(str);
  sftensor input1 = std::make_shared<ftensor>(3, 224, 224);
  input1->Fill(2.f);
  sftensor input2 = std::make_shared<ftensor>(3, 224, 224);
  input2->Fill(3.f);

  sftensor input3 = std::make_shared<ftensor>(3, 224, 224);
  input3->Fill(4.f);

  sftensor input4 = std::make_shared<ftensor>(3, 224, 224);
  input4->Fill(4.f);

  std::vector<sftensor> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

  std::vector<sftensor> outputs(1);
  outputs.at(0) = std::make_shared<ftensor>(3, 224, 224);
  const auto status = expression.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  sftensor output2 = std::make_shared<ftensor>(3, 224, 224);
  output2->Fill(96.f);
  sftensor output1 = outputs.front();

  ASSERT_TRUE(
      arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex5) {
  const std::string &str = "mul(mul(@0,@1), mul(@2,add(@3,@4)))";
  Expression expression(str);
  sftensor input1 = std::make_shared<ftensor>(3, 224, 224);
  input1->Fill(1.f);
  sftensor input2 = std::make_shared<ftensor>(3, 224, 224);
  input2->Fill(2.f);

  sftensor input3 = std::make_shared<ftensor>(3, 224, 224);
  input3->Fill(3.f);

  sftensor input4 = std::make_shared<ftensor>(3, 224, 224);
  input4->Fill(4.f);

  sftensor input5 = std::make_shared<ftensor>(3, 224, 224);
  input5->Fill(5.f);

  std::vector<sftensor> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);
  inputs.push_back(input5);

  std::vector<sftensor> outputs(1);
  outputs.at(0) = std::make_shared<ftensor>(3, 224, 224);
  const auto status = expression.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  sftensor output2 = std::make_shared<ftensor>(3, 224, 224);
  output2->Fill(54.f);
  sftensor output1 = outputs.front();

  ASSERT_TRUE(
      arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_layer, complex6) {
  const std::string &str = "mul(mul(@0,@1), mul(@2,mul(@3,@4)))";
  Expression expression(str);
  sftensor input1 = std::make_shared<ftensor>(3, 224, 224);
  input1->Fill(1.f);
  sftensor input2 = std::make_shared<ftensor>(3, 224, 224);
  input2->Fill(2.f);

  sftensor input3 = std::make_shared<ftensor>(3, 224, 224);
  input3->Fill(3.f);

  sftensor input4 = std::make_shared<ftensor>(3, 224, 224);
  input4->Fill(4.f);

  sftensor input5 = std::make_shared<ftensor>(3, 224, 224);
  input5->Fill(5.f);

  std::vector<sftensor> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);
  inputs.push_back(input5);

  std::vector<sftensor> outputs(1);
  outputs.at(0) = std::make_shared<ftensor>(3, 224, 224);
  const auto status = expression.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  sftensor output2 = std::make_shared<ftensor>(3, 224, 224);
  output2->Fill(120.f);
  sftensor output1 = outputs.front();

  ASSERT_TRUE(
      arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}

TEST(test_parser, tokenize) {
  const std::string &str = "add(add(add(@0,@1),@1),add(@0,@2))";
  ExprParser parser(str);
  parser.Tokenize();
  const auto &tokens = parser.Tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto &token_strs = parser.TokenStrs();
  ASSERT_EQ(token_strs.empty(), false);

  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(tokens.at(0).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(tokens.at(1).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(2), "add");
  ASSERT_EQ(tokens.at(2).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(3), "(");
  ASSERT_EQ(tokens.at(3).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(4), "add");
  ASSERT_EQ(tokens.at(4).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(5), "(");
  ASSERT_EQ(tokens.at(5).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(6), "@0");
  ASSERT_EQ(tokens.at(6).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(7), ",");
  ASSERT_EQ(tokens.at(7).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(8), "@1");
  ASSERT_EQ(tokens.at(8).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(9), ")");
  ASSERT_EQ(tokens.at(9).type, TokenType::TokenRBracket);

  ASSERT_EQ(token_strs.at(10), ",");
  ASSERT_EQ(tokens.at(10).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(11), "@1");
  ASSERT_EQ(tokens.at(11).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(12), ")");
  ASSERT_EQ(tokens.at(12).type, TokenType::TokenRBracket);

  ASSERT_EQ(token_strs.at(13), ",");
  ASSERT_EQ(tokens.at(13).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(14), "add");
  ASSERT_EQ(tokens.at(14).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(15), "(");
  ASSERT_EQ(tokens.at(15).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(16), "@0");
  ASSERT_EQ(tokens.at(16).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(17), ",");
  ASSERT_EQ(tokens.at(17).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(18), "@2");
  ASSERT_EQ(tokens.at(18).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(19), ")");
  ASSERT_EQ(tokens.at(19).type, TokenType::TokenRBracket);
  ASSERT_EQ(token_strs.at(20), ")");
  ASSERT_EQ(tokens.at(20).type, TokenType::TokenRBracket);
}

TEST(test_parser, tokenize2) {
  const std::string &str = "add(add(add(@0,@1),@1),add(@0,add(@1,@1)))";
  ExprParser parser(str);
  parser.Tokenize();
  const auto &tokens = parser.Tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto &token_strs = parser.TokenStrs();
  ASSERT_EQ(token_strs.empty(), false);

  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(tokens.at(0).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(tokens.at(1).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(2), "add");
  ASSERT_EQ(tokens.at(2).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(3), "(");
  ASSERT_EQ(tokens.at(3).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(4), "add");
  ASSERT_EQ(tokens.at(4).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(5), "(");
  ASSERT_EQ(tokens.at(5).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(6), "@0");
  ASSERT_EQ(tokens.at(6).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(7), ",");
  ASSERT_EQ(tokens.at(7).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(8), "@1");
  ASSERT_EQ(tokens.at(8).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(9), ")");
  ASSERT_EQ(tokens.at(9).type, TokenType::TokenRBracket);

  ASSERT_EQ(token_strs.at(10), ",");
  ASSERT_EQ(tokens.at(10).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(11), "@1");
  ASSERT_EQ(tokens.at(11).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(12), ")");
  ASSERT_EQ(tokens.at(12).type, TokenType::TokenRBracket);

  ASSERT_EQ(token_strs.at(13), ",");
  ASSERT_EQ(tokens.at(13).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(14), "add");
  ASSERT_EQ(tokens.at(14).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(15), "(");
  ASSERT_EQ(tokens.at(15).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(16), "@0");
  ASSERT_EQ(tokens.at(16).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(17), ",");
  ASSERT_EQ(tokens.at(17).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(18), "add");
  ASSERT_EQ(tokens.at(18).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(19), "(");
  ASSERT_EQ(tokens.at(19).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(20), "@1");
  ASSERT_EQ(tokens.at(20).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(21), ",");
  ASSERT_EQ(tokens.at(21).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(22), "@1");
  ASSERT_EQ(tokens.at(22).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(23), ")");
  ASSERT_EQ(tokens.at(23).type, TokenType::TokenRBracket);

  ASSERT_EQ(token_strs.at(24), ")");
  ASSERT_EQ(tokens.at(24).type, TokenType::TokenRBracket);

  ASSERT_EQ(token_strs.at(25), ")");
  ASSERT_EQ(tokens.at(25).type, TokenType::TokenRBracket);
}

TEST(test_parser, tokenize3) {
  const std::string &str = "add(add(add(@0,@1),@2),mul(@0,@2))";
  ExprParser parser(str);
  parser.Tokenize();
  const auto &tokens = parser.Tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto &token_strs = parser.TokenStrs();
  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(tokens.at(0).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(tokens.at(1).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(2), "add");
  ASSERT_EQ(tokens.at(2).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(3), "(");
  ASSERT_EQ(tokens.at(3).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(4), "add");
  ASSERT_EQ(tokens.at(4).type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(5), "(");
  ASSERT_EQ(tokens.at(5).type, TokenType::TokenLBracket);

  ASSERT_EQ(token_strs.at(6), "@0");
  ASSERT_EQ(tokens.at(6).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(7), ",");
  ASSERT_EQ(tokens.at(7).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(8), "@1");
  ASSERT_EQ(tokens.at(8).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(9), ")");
  ASSERT_EQ(tokens.at(9).type, TokenType::TokenRBracket);

  ASSERT_EQ(token_strs.at(10), ",");
  ASSERT_EQ(tokens.at(10).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(11), "@2");
  ASSERT_EQ(tokens.at(11).type, TokenType::TokenInputNum);

  ASSERT_EQ(token_strs.at(12), ")");
  ASSERT_EQ(tokens.at(12).type, TokenType::TokenRBracket);

  ASSERT_EQ(token_strs.at(13), ",");
  ASSERT_EQ(tokens.at(13).type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(14), "mul");
  ASSERT_EQ(tokens.at(14).type, TokenType::TokenMul);

  ASSERT_EQ(token_strs.at(15), "(");
  ASSERT_EQ(tokens.at(15).type, TokenType::TokenLBracket);
}

TEST(test_parser, generate1) {
  const std::string &str = "add(@0,@1)";
  ExprParser parser(str);
  parser.Tokenize();
  const auto &nodes = parser.Generate();
  ASSERT_EQ(nodes.size(), 3);
  ASSERT_EQ(nodes.at(0)->num, 0);
  ASSERT_EQ(nodes.at(1)->num, 1);
  ASSERT_EQ(nodes.at(2)->num, int(TokenType::TokenAdd));
}

TEST(test_parser, generate2) {
  const std::string &str = "add(@0,add(@1,@2))";
  ExprParser parser(str);
  parser.Tokenize();
  const auto &nodes = parser.Generate();
  ASSERT_EQ(nodes.size(), 5);
  ASSERT_EQ(nodes.at(0)->num, 0);
  ASSERT_EQ(nodes.at(1)->num, 1);
  ASSERT_EQ(nodes.at(2)->num, 2);

  ASSERT_EQ(nodes.at(3)->num, int(TokenType::TokenAdd));
  ASSERT_EQ(nodes.at(4)->num, int(TokenType::TokenAdd));
}

TEST(test_parser, generate3) {
  const std::string &str = "add(@0,add(@1,add(@3,@4)))";
  ExprParser parser(str);
  parser.Tokenize();
  const auto &nodes = parser.Generate();
  ASSERT_EQ(nodes.size(), 7);
  ASSERT_EQ(nodes.at(0)->num, 0);
  ASSERT_EQ(nodes.at(1)->num, 1);
  ASSERT_EQ(nodes.at(2)->num, 3);

  ASSERT_EQ(nodes.at(3)->num, 4);
  ASSERT_EQ(nodes.at(4)->num, int(TokenType::TokenAdd));
  ASSERT_EQ(nodes.at(5)->num, int(TokenType::TokenAdd));
  ASSERT_EQ(nodes.at(6)->num, int(TokenType::TokenAdd));
}

TEST(test_parser, generate4) {
  const std::string &str = "add(@0,add(@1,add(@3,mul(@4,@5))))";
  ExprParser parser(str);
  parser.Tokenize();
  const auto &nodes = parser.Generate();
  ASSERT_EQ(nodes.size(), 9);
  ASSERT_EQ(nodes.at(0)->num, 0);
  ASSERT_EQ(nodes.at(1)->num, 1);
  ASSERT_EQ(nodes.at(2)->num, 3);
  ASSERT_EQ(nodes.at(3)->num, 4);
  ASSERT_EQ(nodes.at(4)->num, 5);
  ASSERT_EQ(nodes.at(5)->num, int(TokenType::TokenMul));
  ASSERT_EQ(nodes.at(6)->num, int(TokenType::TokenAdd));
  ASSERT_EQ(nodes.at(7)->num, int(TokenType::TokenAdd));
  ASSERT_EQ(nodes.at(8)->num, int(TokenType::TokenAdd));
}