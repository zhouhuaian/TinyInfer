#include <gtest/gtest.h>
#include "layer/abstract/attr_layer.hpp"

using namespace TinyInfer;

TEST(test_attr_layer, set_weights1) {
  AttrLayer attr_layer("attr");
  uint32_t weight_count = 4;
  std::vector<sftensor> weights;
  weights.reserve(weight_count);
  for (uint32_t i = 0; i < weight_count; ++i) {
    sftensor weight = std::make_shared<ftensor>(3, 32, 32);
    weights.push_back(weight);
  }

  attr_layer.set_weights(weights);
  const auto& weights_ = attr_layer.weights();
  ASSERT_EQ(weights.size(), weights_.size());

  for (uint32_t i = 0; i < weight_count; ++i) {
    const auto& weight_ = weights_.at(i);
    const auto& weight = weights.at(i);
    ASSERT_EQ(weight->size(), weight_->size());
    
    for (uint32_t j = 0; j < weight->size(); ++j) {
      ASSERT_EQ(weight->index(j), weight_->index(j));
    }
  }
}

TEST(test_attr_layer, set_bias1) {
  AttrLayer attr_layer("attr");
  uint32_t bias_count = 4;
  std::vector<sftensor> biases;
  biases.reserve((bias_count));
  for (uint32_t i = 0; i < bias_count; ++i) {
    sftensor bias = std::make_shared<ftensor>(1, 32, 1);
    biases.push_back(bias);
  }
  attr_layer.set_bias(biases);
  const auto& biases_ = attr_layer.bias();
  ASSERT_EQ(biases.size(), biases_.size());

  for (uint32_t i = 0; i < bias_count; ++i) {
    const auto& bias_ = biases_.at(i);
    const auto& bias = biases.at(i);
    ASSERT_EQ(bias->size(), bias_->size());
    
    for (uint32_t j = 0; j < bias->size(); ++j) {
      ASSERT_EQ(bias->index(j), bias_->index(j));
    }
  }
}

TEST(test_attr_layer, set_weights2) {
  AttrLayer attr_layer("attr");
  uint32_t val_count = 9;
  std::vector<float> weight_vals;
  weight_vals.reserve(val_count);
  for (int i = 0; i < val_count; ++i) {
    weight_vals.push_back(float(i));
  }
  std::vector<sftensor> weights_;
  sftensor weight = std::make_shared<ftensor>(1, 3, 3);
  weights_.push_back(weight);
  attr_layer.set_weights(weights_);
  attr_layer.set_weights(weight_vals);
  
  weights_ = attr_layer.weights();
  for (int i = 0; i < weights_.size(); ++i) {
    const auto weight_ = weights_.at(i);
    int idx = 0;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        ASSERT_EQ(weight_->at(0, r, c), idx);
        idx += 1;
      }
    }
  }
}

TEST(test_attr_layer, set_bias2) {
  AttrLayer attr_layer("attr");
  uint32_t val_count = 9;
  std::vector<float> bias_vals;
  bias_vals.reserve(val_count);
  for (int i = 0; i < val_count; ++i) {
    bias_vals.push_back(float(i));
  }
  std::vector<sftensor> biases_;
  sftensor bias = std::make_shared<ftensor>(1, 9, 1);
  biases_.push_back(bias);
  attr_layer.set_bias(biases_);
  attr_layer.set_bias(bias_vals);
  
  biases_ = attr_layer.bias();
  for (int i = 0; i < biases_.size(); ++i) {
    const auto bias_ = biases_.at(i);
    int idx = 0;
    for (int r = 0; r < 9; ++r) {
      ASSERT_EQ(bias_->at(0, r, 0), idx);
      idx += 1;
    }
  }
}

TEST(test_attr_layer, init_weights) {
  AttrLayer attr_layer("attr");
  attr_layer.InitWeights(3, 64, 32, 32);
  const std::vector<sftensor> weights = attr_layer.weights();
  ASSERT_EQ(weights.size(), 3);
  for (uint32_t i = 0; i < weights.size(); ++i) {
    ASSERT_EQ(weights.at(i)->channels(), 64);
    ASSERT_EQ(weights.at(i)->rows(), 32);
    ASSERT_EQ(weights.at(i)->cols(), 32);
  }
}

TEST(test_attr_layer, init_bias) {
  AttrLayer attr_layer("attr");
  attr_layer.InitBias(3, 64, 1, 1);
  const std::vector<sftensor> bias = attr_layer.bias();
  ASSERT_EQ(bias.size(), 3);
  for (uint32_t i = 0; i < bias.size(); ++i) {
    ASSERT_EQ(bias.at(i)->channels(), 64);
    ASSERT_EQ(bias.at(i)->rows(), 1);
    ASSERT_EQ(bias.at(i)->cols(), 1);
  }
}
