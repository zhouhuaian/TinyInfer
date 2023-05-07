#include "../../src/kernel/details/linear.hpp"
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace TinyInfer;

TEST(test_kernel, forward_linear1) {
  const uint32_t in_features = 32;
  const uint32_t out_features = 64;
  const uint32_t in_dims = 1280;

  Linear linear(in_features, out_features, false);
  std::vector<float> weights(in_features * out_features, 1.f);
  linear.set_weights(weights);

  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill(1.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), in_features);
    }
  }
}

TEST(test_kernel, forward_linear2) {
  const uint32_t in_features = 32;
  const uint32_t out_features = 64;
  const uint32_t in_dims = 1280;

  Linear linear(in_features, out_features, false);
  std::vector<float> weights(in_features * out_features, 1.f);
  linear.set_weights(weights);

  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill(2.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), in_features * 2.f);
    }
  }
}

TEST(test_kernel, forward_linear3) {
  const uint32_t in_features = 8;
  const uint32_t out_features = 12;
  const uint32_t in_dims = 4;

  Linear linear(in_features, out_features, false);

  std::vector<float> weights_raw;
  weights_raw.reserve(out_features * in_features);
  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }
  linear.set_weights(weights_raw);

  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill(1.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), 36.f);
    }
  }
}

TEST(test_kernel, forward_linear4) {
  const uint32_t in_features = 64;
  const uint32_t out_features = 128;
  const uint32_t in_dims = 4;

  Linear linear(in_features, out_features, false);

  std::vector<float> weights_raw;
  weights_raw.reserve(out_features * in_features);
  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }
  linear.set_weights(weights_raw);

  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill(1.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), 2080.f);
    }
  }
}

TEST(test_kernel, forward_linear5) {
  const uint32_t in_features = 64;
  const uint32_t out_features = 128;
  const uint32_t in_dims = 4;

  Linear linear(in_features, out_features, false);

  std::vector<float> weights_raw;
  weights_raw.reserve(out_features * in_features);
  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }
  linear.set_weights(weights_raw);

  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill(2.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), 2080 * 2.f);
    }
  }
}

TEST(test_kernel, forward_linear6) {
  const uint32_t in_features = 2;
  const uint32_t out_features = 4;
  const uint32_t in_dims = 3;

  Linear linear(in_features, out_features, false);

  std::vector<float> weights_raw;
  weights_raw.reserve(out_features * in_features);
  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }
  linear.set_weights(weights_raw);

  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill({1, 2, 3, 4, 5, 6}, true);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  const auto &result = outputs.front();
  for (int i = 0; i < out_features; ++i) {
    ASSERT_EQ(result->at(0, i, 0), 9.f);
    ASSERT_EQ(result->at(0, i, 1), 12.f);
    ASSERT_EQ(result->at(0, i, 2), 15.f);
  }
}

TEST(test_kernel, forward_linear7) {
  const uint32_t in_features = 3;
  const uint32_t out_features = 4;
  const uint32_t in_dims = 3;

  Linear linear(in_features, out_features, false);

  std::vector<float> weights_raw;
  weights_raw.reserve(out_features * in_features);
  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }
  linear.set_weights(weights_raw);

  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill({1, 2, 3, 4, 5, 6, 7, 8, 9}, true);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  const auto &result = outputs.front();
  for (int i = 0; i < out_features; ++i) {
    ASSERT_EQ(result->at(0, i, 0), 30.f);
    ASSERT_EQ(result->at(0, i, 1), 36.f);
    ASSERT_EQ(result->at(0, i, 2), 42.f);
  }
}

TEST(test_kernel, forward_linear8) {
  const uint32_t in_features = 3;
  const uint32_t out_features = 5;
  const uint32_t in_dims = 4;

  Linear linear(in_features, out_features, false);

  std::vector<float> weights_raw;
  weights_raw.reserve(out_features * in_features);
  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }
  linear.set_weights(weights_raw);

  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill({1, 2, 3, 13, 4, 5, 6, 15, 7, 8, 9, 16}, true);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  const auto &result = outputs.front();
  for (int i = 0; i < out_features; ++i) {
    ASSERT_EQ(result->at(0, i, 0), 30.f);
    ASSERT_EQ(result->at(0, i, 1), 36.f);
    ASSERT_EQ(result->at(0, i, 2), 42.f);
    ASSERT_EQ(result->at(0, i, 3), 91.f);
  }
}

TEST(test_kernel, forward_linear9) {
  const uint32_t in_features = 32;
  const uint32_t out_features = 48;
  const uint32_t in_dims = 4;
  Linear linear(in_features, out_features, false);

  std::vector<float> weights_raw;
  weights_raw.reserve(out_features * in_features);

  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(1.f);
    }
  }
  linear.set_weights(weights_raw);

  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill(1.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  const auto &result = outputs.front();
  for (int i = 0; i < out_features; ++i) {
    ASSERT_EQ(result->at(0, i, 0), 32.f);
    ASSERT_EQ(result->at(0, i, 1), 32.f);
    ASSERT_EQ(result->at(0, i, 2), 32.f);
    ASSERT_EQ(result->at(0, i, 3), 32.f);
  }
}

TEST(test_kernel, forward_linear10) {
  const uint32_t in_features = 32;
  const uint32_t out_features = 48;
  const uint32_t in_dims = 4;
  Linear linear(in_features, out_features, false);

  std::vector<float> weights_raw;
  weights_raw.reserve(out_features * in_features);

  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(1.f);
    }
  }
  linear.set_weights(weights_raw);

  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill(1.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  const auto &result = outputs.front();
  for (int i = 0; i < out_features; ++i) {
    ASSERT_EQ(result->at(0, i, 0), 32.f);
    ASSERT_EQ(result->at(0, i, 1), 32.f);
    ASSERT_EQ(result->at(0, i, 2), 32.f);
    ASSERT_EQ(result->at(0, i, 3), 32.f);
  }
}

TEST(test_kernel, forward_linear11) {
  const uint32_t in_features = 32;
  const uint32_t out_features = 48;
  const uint32_t in_dims = 4;
  Linear linear(in_features, out_features, false);

  std::vector<float> weights_raw;
  weights_raw.reserve(out_features * in_features);
  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(1.f);
    }
  }
  linear.set_weights(weights_raw);

  std::vector<float> input_raw;
  input_raw.reserve(in_features * in_dims);
  float val = 1.f;
  for (int i = 0; i < in_features; ++i) {
    for (int j = 0; j < in_dims; ++j) {
      input_raw.push_back(val);
      val += 1;
    }
  }
  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill(input_raw, true);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  const auto &result = outputs.front();
  for (int i = 0; i < out_features; ++i) {
    ASSERT_EQ(result->at(0, i, 0), 2016.f);
    ASSERT_EQ(result->at(0, i, 1), 2048.f);
    ASSERT_EQ(result->at(0, i, 2), 2080.f);
    ASSERT_EQ(result->at(0, i, 3), 2112.f);
  }
}

TEST(test_kernel, forward_linear12) {
  const uint32_t in_features = 32;
  const uint32_t out_features = 96;
  const uint32_t in_dims = 5;
  Linear linear(in_features, out_features, false);

  std::vector<float> weights_raw;
  weights_raw.reserve(out_features * in_features);
  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(1.f);
    }
  }
  linear.set_weights(weights_raw);

  std::vector<float> input_raw;
  input_raw.reserve(in_features * in_dims);
  float val = 1.f;
  for (int i = 0; i < in_features; ++i) {
    for (int j = 0; j < in_dims; ++j) {
      input_raw.push_back(val);
      val += 1;
    }
  }
  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill(input_raw, true);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  const auto status = linear.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  const auto &result = outputs.front();
  for (int i = 0; i < out_features; ++i) {
    ASSERT_EQ(result->at(0, i, 0), 2512.f);
    ASSERT_EQ(result->at(0, i, 1), 2544.f);
    ASSERT_EQ(result->at(0, i, 2), 2576.f);
    ASSERT_EQ(result->at(0, i, 3), 2608.f);
    ASSERT_EQ(result->at(0, i, 4), 2640.f);
  }
}
