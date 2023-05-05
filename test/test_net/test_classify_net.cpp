#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace TinyInfer;

TEST(test_net, forward_resnet18) {
  RuntimeGraph graph("../../tmp/resnet/resnet18_batch1.param",
                     "../../tmp/resnet/resnet18_batch1.pnnx.bin");
  graph.Build("pnnx_input_0", "pnnx_output_0");

  int repeat_number = 2;
  for (int r = 0; r < repeat_number; ++r) {
    sftensor input1 = std::make_shared<ftensor>(3, 224, 224);
    input1->Fill(2.);

    std::vector<sftensor> inputs;
    inputs.push_back(input1);

    std::vector<sftensor> outputs = graph.Forward(inputs, false);
    ASSERT_EQ(outputs.size(), 1);

    const auto &output1 = outputs.front()->slice(0);
    const auto &output2 = CSVDataLoader::LoadData("../../tmp/resnet/1.csv");
    ASSERT_EQ(output1.size(), output2.size());
    for (uint32_t i = 0; i < output1.size(); ++i) {
      ASSERT_LE(std::abs(output1.at(i) - output2.at(i)), 5e-6);
    }
  }
}

TEST(test_net, forward_group_conv) {
  RuntimeGraph graph("../../tmp/group_conv/group_conv.pnnx.param",
                     "../../tmp/group_conv/group_conv.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  sftensor input1 = std::make_shared<ftensor>(4, 16, 16);
  input1->Fill(2.f);

  std::vector<sftensor> inputs;
  inputs.push_back(input1);

  std::vector<sftensor> outputs = graph.Forward(inputs, false);
  ASSERT_EQ(outputs.size(), 1);

  const auto &output1 = outputs.front()->slice(0);
  const auto &output2 = CSVDataLoader::LoadData("../../tmp/mobilenet/1.csv");
  ASSERT_EQ(output1.size(), output2.size());
  for (uint32_t i = 0; i < output1.size(); ++i) {
    ASSERT_LE(std::abs(output1.at(i) - output2.at(i)), 5e-6);
  }
}

TEST(test_net, forward_mobilenet1) {
  RuntimeGraph graph("../../tmp/mobilenet/mobile.pnnx.param",
                     "../../tmp/mobilenet/mobile.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  sftensor input1 = std::make_shared<ftensor>(3, 32, 32);
  input1->Fill(1.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input1);

  int repeat_size = 3;
  for (int r = 0; r < repeat_size; ++r) {
    std::vector<sftensor> outputs = graph.Forward(inputs, false);
    ASSERT_EQ(outputs.size(), 1);

    const auto &output1 = outputs.front()->data();
    const auto &output2 = CSVDataLoader::LoadData("../../tmp/mobilenet/2.csv");
    ASSERT_EQ(output1.size(), output2.size());
    for (uint32_t i = 0; i < output1.size(); ++i) {
      ASSERT_LE(std::abs(output1.at(i) - output2.at(i)), 5e-6);
    }
  }
}

TEST(test_net, forward_mobilenet2) {
  RuntimeGraph graph("../../tmp/mobilenet/mobile_224.pnnx.param",
                     "../../tmp/mobilenet/mobile_224.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t channels = 3;
  sftensor input1 = std::make_shared<ftensor>(channels, 224, 224);
  input1->Fill(1.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input1);

  int repeat_size = 3;
  for (int r = 0; r < repeat_size; ++r) {
    std::vector<sftensor> outputs = graph.Forward(inputs, false);
    ASSERT_EQ(outputs.size(), 1);

    const auto &output1 = outputs.front()->data();
    const auto &output2 = CSVDataLoader::LoadData("../../tmp/mobilenet/3.csv");
    ASSERT_EQ(output1.size(), output2.size());
    for (uint32_t i = 0; i < output1.size(); ++i) {
      ASSERT_LE(std::abs(output1.at(i) - output2.at(i)), 5e-6);
    }
  }
}
