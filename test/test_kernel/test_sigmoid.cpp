#include <glog/logging.h>
#include <gtest/gtest.h>

#include "../../src/kernel/details/sigmoid.hpp"
#include "data/tensor.hpp"

using namespace TinyInfer;

TEST(test_kernel, forward_sigmoid1) {
  sftensor input = std::make_shared<ftensor>(1, 1, 4);
  input->index(0) = 1.f;
  input->index(1) = 2.f;
  input->index(2) = 3.f;
  input->index(3) = 4.f;

  std::vector<sftensor> inputs;
  inputs.push_back(input);

  std::vector<sftensor> outputs;
  sftensor output = std::make_shared<ftensor>(1, 1, 4);
  outputs.push_back(output);

  Sigmoid sigmoid;
  const auto status = sigmoid.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_LE(std::abs(outputs.front()->index(0) - 0.7310585786300049f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(1) - 0.8807970779778823f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(2) - 0.9525741268224334f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(3) - 0.9820137900379085f), 1e-6);
}

TEST(test_kernel, forward_sigmoid2) {
  sftensor input = std::make_shared<ftensor>(1, 1, 4);
  input->index(0) = 11.f;
  input->index(1) = 22.f;
  input->index(2) = 33.f;
  input->index(3) = 44.f;

  std::vector<sftensor> inputs;
  inputs.push_back(input);

  std::vector<sftensor> outputs;
  sftensor output = std::make_shared<ftensor>(1, 1, 4);
  outputs.push_back(output);

  Sigmoid sigmoid;
  const auto status = sigmoid.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_LE(std::abs(outputs.front()->index(0) - 0.999983298578152f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(1) - 0.9999999997210531f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(2) - 0.9999999999999953f), 1e-6);
  ASSERT_LE(std::abs(outputs.front()->index(3) - 1.0f), 1e-6);
}

TEST(test_kernel, forward_sigmoid3) {
  sftensor input = std::make_shared<ftensor>(32, 224, 512);
  input->Rand();
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(inputs.size());

  Sigmoid sigmoid;
  const auto status = sigmoid.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  for (int b = 0; b < inputs.size(); ++b) {
    sftensor input_ = inputs.at(b);
    sftensor output_ = outputs.at(b);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_LE(output_->index(i) - 1.f / (1.f + std::exp(-input_->index(i))),
                1e-6);
    }
  }
}

TEST(test_kernel, forward_sigmoid4) {
  sftensor input = std::make_shared<ftensor>(1, 32, 128);
  input->Rand();
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(inputs.size());

  Sigmoid sigmoid;
  const auto status = sigmoid.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  for (int b = 0; b < inputs.size(); ++b) {
    sftensor input_ = inputs.at(b);
    sftensor output_ = outputs.at(b);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_LE(output_->index(i) - 1.f / (1.f + std::exp(-input_->index(i))),
                1e-6);
    }
  }
}

TEST(test_kernel, forward_sigmoid5) {
  sftensor input = std::make_shared<ftensor>(1, 1, 128);
  input->Rand();
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(inputs.size());

  Sigmoid sigmoid;
  const auto status = sigmoid.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  for (int b = 0; b < inputs.size(); ++b) {
    sftensor input_ = inputs.at(b);
    sftensor output_ = outputs.at(b);
    CHECK(input_->size() == output_->size());
    uint32_t size = input_->size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_LE(output_->index(i) - 1.f / (1.f + std::exp(-input_->index(i))),
                1e-6);
    }
  }
}