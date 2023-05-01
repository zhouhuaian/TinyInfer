#include <gtest/gtest.h>
#include <glog/logging.h>

#include "data/tensor.hpp"
#include "../../src/layer/details/hardsigmoid.hpp"

using namespace TinyInfer;

TEST(test_layer, forward_hardsigmoid1) {
  sftensor input = std::make_shared<ftensor>(32, 224, 512);
  input->Rand();
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(inputs.size());

  HardSigmoid hardsigmoid_layer;
  const auto status = hardsigmoid_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  for (int b = 0; b < inputs.size(); ++b) {
    sftensor input_ = inputs.at(b);
    sftensor output_ = outputs.at(b);
    CHECK(input_->size() == output_->size());
    input_->Transform([] (float val) {
      if (val <= -3.f) {
        return 0.f;
      } else if (val >= 3.f) {
        return 1.f;
      } else {
        return val / 6.f + 0.5f;
      }
    });
    uint32_t size = input_->size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_EQ(output_->index(i), input_->index(i));
    }
  }
}

TEST(test_layer, forward_hardsigmoid2) {
  sftensor input = std::make_shared<ftensor>(1, 32, 128);
  input->Rand();
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(inputs.size());

  HardSigmoid hardsigmoid_layer;
  const auto status = hardsigmoid_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  for (int b = 0; b < inputs.size(); ++b) {
    sftensor input_ = inputs.at(b);
    sftensor output_ = outputs.at(b);
    CHECK(input_->size() == output_->size());
    input_->Transform([] (float val) {
      if (val <= -3.f) {
        return 0.f;
      } else if (val >= 3.f) {
        return 1.f;
      } else {
        return val / 6.f + 0.5f;
      }
    });
    uint32_t size = input_->size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_EQ(output_->index(i), input_->index(i));
    }
  }
}

TEST(test_layer,  forward_hardsigmoid3) {
  sftensor input = std::make_shared<ftensor>(1, 1, 16);
  input->Rand();
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(inputs.size());

  HardSigmoid hardsigmoid_layer;
  const auto status = hardsigmoid_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  for (int b = 0; b < inputs.size(); ++b) {
    sftensor input_ = inputs.at(b);
    sftensor output_ = outputs.at(b);
    CHECK(input_->size() == output_->size());
    input_->Transform([] (float val) {
      if (val <= -3.f) {
        return 0.f;
      } else if (val >= 3.f) {
        return 1.f;
      } else {
        return val / 6.f + 0.5f;
      }
    });
    uint32_t size = input_->size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_EQ(output_->index(i), input_->index(i));
    }
  }
}