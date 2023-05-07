#include "../src/kernel/details/convolution.hpp"
#include "runtime/runtime_graph.hpp"
#include <benchmark/benchmark.h>

using namespace TinyInfer;

static void BM_Convolutionk3x3s1x1(benchmark::State &state) {
  uint32_t kernel_ct = state.range(0);
  uint32_t channels = state.range(1);
  uint32_t rows = state.range(2);
  uint32_t cols = state.range(3);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  std::vector<sftensor> outputs(1);

  std::vector<sftensor> weights(kernel_ct);
  for (uint32_t k = 0; k < kernel_ct; ++k) {
    auto &weight = weights.at(k);
    weight = std::make_shared<ftensor>(channels, 3, 3);
    weight->Rand();
  }

  Convolution convolution(kernel_ct, channels, 3, 3, 0, 0, 1, 1, 1, false);
  convolution.set_weights(weights);

  for (auto _ : state) {
    convolution.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Convolutionk3x3s1x1)
    ->Args({32, 3, 320, 320})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolutionk3x3s1x1)
    ->Args({64, 32, 160, 160})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolutionk3x3s1x1)
    ->Args({128, 64, 80, 80})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolutionk3x3s1x1)
    ->Args({256, 128, 40, 40})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolutionk3x3s1x1)
    ->Args({512, 256, 20, 20})
    ->Unit(benchmark::kMillisecond);
