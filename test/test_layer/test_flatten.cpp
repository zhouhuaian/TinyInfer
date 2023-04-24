#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../../src/layer/details/flatten.hpp"

using namespace TinyInfer;

TEST(test_layer, forward_flatten_layer1) {
  std::vector<sftensor> inputs;
  sftensor input = std::make_shared<ftensor>(8, 24, 32);
  input->Rand();
  inputs.push_back(input);

  std::vector<sftensor> outputs(inputs.size());

  Flatten flatten(1, 3);
  const auto status = flatten.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);

  const auto& shapes = outputs.front()->shape();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24 * 32);
  ASSERT_EQ(shapes.at(2), 1);

  const auto& raw_shapes = outputs.front()->raw_shape();
  ASSERT_EQ(raw_shapes.size(), 1);
  ASSERT_EQ(raw_shapes.front(), 8 * 24 * 32);

  uint32_t batch1 = inputs.front()->size();
  uint32_t batch2 = outputs.front()->size();
  ASSERT_EQ(batch1, batch2);
}

TEST(test_layer, forward_flatten_layer2) {
  std::vector<sftensor> inputs;
  sftensor input = std::make_shared<ftensor>(8, 24, 32);
  input->Rand();
  inputs.push_back(input);

  std::vector<sftensor> outputs(inputs.size());

  Flatten flatten(1, -1);
  const auto status = flatten.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);
  ASSERT_EQ(outputs.size(), 1);

  const auto& shapes = outputs.front()->shape();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24 * 32);
  ASSERT_EQ(shapes.at(2), 1);

  const auto& raw_shapes = outputs.front()->raw_shape();
  ASSERT_EQ(raw_shapes.size(), 1);

  ASSERT_EQ(raw_shapes.front(), 8 * 24 * 32);

  uint32_t batch1 = inputs.front()->size();
  uint32_t batch2 = outputs.front()->size();
  ASSERT_EQ(batch1, batch2);
}

TEST(test_layer, forward_flatten_layer3) {
  std::vector<sftensor> inputs;
  sftensor input = std::make_shared<ftensor>(8, 24, 32);
  input->Rand();
  inputs.push_back(input);

  std::vector<sftensor> outputs(inputs.size());

  Flatten flatten(1, 2);
  const auto status = flatten.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);;
  ASSERT_EQ(outputs.size(), 1);

  const auto& shapes = outputs.front()->shape();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24);
  ASSERT_EQ(shapes.at(2), 32);

  const auto& raw_shapes = outputs.front()->raw_shape();
  ASSERT_EQ(raw_shapes.size(), 2);

  ASSERT_EQ(raw_shapes.at(0), 8 * 24);
  ASSERT_EQ(raw_shapes.at(1), 32);

  uint32_t batch1 = inputs.front()->size();
  uint32_t batch2 = outputs.front()->size();
  ASSERT_EQ(batch1, batch2);

}

TEST(test_layer, forward_flatten_layer4) {
  std::vector<sftensor> inputs;
  sftensor input = std::make_shared<ftensor>(8, 24, 32);
  input->Rand();
  inputs.push_back(input);

  std::vector<sftensor> outputs(inputs.size());

  Flatten flatten(1, -2);
  const auto status = flatten.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);

  ASSERT_EQ(outputs.size(), 1);

  const auto& shapes = outputs.front()->shape();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8 * 24);
  ASSERT_EQ(shapes.at(2), 32);

  const auto& raw_shapes = outputs.front()->raw_shape();
  ASSERT_EQ(raw_shapes.size(), 2);

  ASSERT_EQ(raw_shapes.at(0), 8 * 24);
  ASSERT_EQ(raw_shapes.at(1), 32);

  uint32_t batch1 = inputs.front()->size();
  uint32_t batch2 = outputs.front()->size();
  ASSERT_EQ(batch1, batch2);
}

TEST(test_layer, forward_flatten_layer5) {
  std::vector<sftensor> inputs;
  sftensor input = std::make_shared<ftensor>(8, 24, 32);
  input->Rand();
  inputs.push_back(input);

  std::vector<sftensor> outputs(inputs.size());

  Flatten flatten(2, 3);
  const auto status = flatten.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);

  ASSERT_EQ(outputs.size(), 1);

  const auto& shapes = outputs.front()->shape();
  ASSERT_EQ(shapes.size(), 3);

  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 8);
  ASSERT_EQ(shapes.at(2), 24 * 32);

  const auto& raw_shapes = outputs.front()->raw_shape();
  ASSERT_EQ(raw_shapes.size(), 2);

  ASSERT_EQ(raw_shapes.at(0), 8);
  ASSERT_EQ(raw_shapes.at(1), 24 * 32);

  uint32_t batch1 = inputs.front()->size();
  uint32_t batch2 = outputs.front()->size();
  ASSERT_EQ(batch1, batch2);
}