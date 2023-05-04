#include <benchmark/benchmark.h>
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"

using namespace TinyInfer;

static void BM_MobilenetV3_Batch8_224x224(benchmark::State& state) {
  RuntimeGraph graph("../../tmp/mobilenet/mobile_batch8.pnnx.param",
                     "../../tmp/mobilenet/mobile_batch8.bin");

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

BENCHMARK(BM_MobilenetV3_Batch8_224x224)->Unit(benchmark::kMillisecond);

