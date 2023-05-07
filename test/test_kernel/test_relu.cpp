#include <glog/logging.h>
#include <gtest/gtest.h>

#include "../../src/kernel/details/relu.hpp"
#include "data/tensor.hpp"

using namespace TinyInfer;

TEST(test_kernel, forward_relu1) {
  sftensor input = std::make_shared<ftensor>(32, 224, 512);
  input->Rand();
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(inputs.size());

  ReLU relu;
  const auto status = relu.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);

  for (int b = 0; b < inputs.size(); ++b) {
    sftensor input_ = inputs.at(b);
    input_->Transform([](const float val) { return val > 0.f ? val : 0.f; });
    sftensor output_ = outputs.at(b);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_EQ(output_->index(i), input_->index(i));
    }
  }
}

TEST(test_kernel, forward_relu2) {
  sftensor input = std::make_shared<ftensor>(1, 32, 128);
  input->Rand();
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(inputs.size());

  ReLU relu;
  const auto status = relu.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);

  for (int b = 0; b < inputs.size(); ++b) {
    sftensor input_ = inputs.at(b);
    input_->Transform([](const float val) { return val > 0.f ? val : 0.f; });
    sftensor output_ = outputs.at(b);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_EQ(output_->index(i), input_->index(i));
    }
  }
}

TEST(test_kernel, forward_relu3) {
  sftensor input = std::make_shared<ftensor>(1, 1, 16);
  input->Rand();
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(inputs.size());

  ReLU relu;
  const auto status = relu.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);

  for (int b = 0; b < inputs.size(); ++b) {
    sftensor input_ = inputs.at(b);
    input_->Transform([](const float val) { return val > 0.f ? val : 0.f; });
    sftensor output_ = outputs.at(b);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_EQ(output_->index(i), input_->index(i));
    }
  }
}

TEST(test_kernel, forward_relu4) {
  sftensor input = std::make_shared<ftensor>(1, 1, 15);
  input->Rand();
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(inputs.size());

  ReLU relu;
  const auto status = relu.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);

  for (int b = 0; b < inputs.size(); ++b) {
    sftensor input_ = inputs.at(b);
    input_->Transform([](const float val) { return val > 0.f ? val : 0.f; });
    sftensor output_ = outputs.at(b);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_EQ(output_->index(i), input_->index(i));
    }
  }
}