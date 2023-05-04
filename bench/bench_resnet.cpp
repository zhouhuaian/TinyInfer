#include <benchmark/benchmark.h>
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"

using namespace TinyInfer;

// const static int kIterationNum = 5;

static void BM_Resnet18_Batch8_224x224(benchmark::State& state) {
  RuntimeGraph graph("../../tmp/resnet/resnet18_batch8.pnnx.param",
                     "../../tmp/resnet/resnet18_batch8.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  
  const uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);
  for (int b = 0; b < batch; ++b) {
    auto& input = inputs.at(b);
    input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(1.f);
  }

  for (auto _ : state) {
    graph.Forward(inputs, false);
  }
}

static void BM_Resnet18_Batch16_224x224(benchmark::State& state) {
  RuntimeGraph graph("../../tmp/resnet/resnet18_batch16.pnnx.param",
                     "../../tmp/resnet/resnet18_batch16.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  const uint32_t batch = 16;
  std::vector<sftensor> inputs(batch);
  for (int b = 0; b < batch; ++b) {
    auto& input = inputs.at(b);
    input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(1.f);
  }

  for (auto _ : state) {
    graph.Forward(inputs, false);
  }
}

BENCHMARK(BM_Resnet18_Batch8_224x224)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Resnet18_Batch16_224x224)->Unit(benchmark::kMillisecond);