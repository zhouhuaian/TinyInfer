#include <benchmark/benchmark.h>
#include <cstdint>
#include <memory>
#include "../src/layer/details/adapt_avgpooling.hpp"
#include "../src/layer/details/cat.hpp"
#include "../src/layer/details/expression.hpp"
#include "../src/layer/details/flatten.hpp"
#include "../src/layer/details/hardsigmoid.hpp"
#include "../src/layer/details/hardswish.hpp"
#include "../src/layer/details/linear.hpp"
#include "../src/layer/details/maxpooling.hpp"
#include "../src/layer/details/relu.hpp"
#include "../src/layer/details/sigmoid.hpp"
#include "../src/layer/details/softmax.hpp"
#include "data/tensor.hpp"

using namespace TinyInfer;

static void BM_Concat16to8(benchmark::State& state) {
  int in_batch = 16;
  int out_batch = 8;
  uint32_t in_channels = state.range(0);
  uint32_t in_rows = state.range(1);
  uint32_t in_columns = state.range(2);  

  std::vector<sftensor> inputs(in_batch);
  for (int b = 0; b < in_batch; ++b) {
    auto& input = inputs.at(b);
    input = std::make_shared<ftensor>(in_channels, in_rows, in_columns);
    input->Rand();
  }

  std::vector<sftensor> outputs(out_batch);
  Cat cat(1);

  for (auto _ : state) {
    cat.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Concat16to8)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Concat16to8)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Concat16to8)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Concat16to8)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);


static void BM_Sigmoid(benchmark::State& state) {

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);  
  for (uint32_t b = 0; b < batch; ++b) {
    auto& input = inputs.at(b);
    input = std::make_shared<ftensor>(channels, rows, cols);
    input->Fill(1.f * b);
  }
  
  std::vector<sftensor> outputs(batch);

  Sigmoid sigmoid;
  for (auto _ : state) {
    sigmoid.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Sigmoid)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Sigmoid)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Sigmoid)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Sigmoid)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);


static void BM_HardSigmoid(benchmark::State& state) {

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);  
  for (uint32_t b = 0; b < batch; ++b) {
    auto& input = inputs.at(b);
    input = std::make_shared<ftensor>(channels, rows, cols);
    input->Fill(1.f * b);
  }
  
  std::vector<sftensor> outputs(batch);

  HardSigmoid hardsigmoid;
  for (auto _ : state) {
    hardsigmoid.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_HardSigmoid)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSigmoid)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSigmoid)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSigmoid)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);


static void BM_Linear(benchmark::State& state) {
  const int32_t in_features = (int32_t)state.range(0);
  const int32_t out_features = (int32_t)state.range(1);
  const int32_t in_dims = (int32_t)state.range(2);

  Linear linear(in_features, out_features, false);
  std::vector<float> weight_vals(in_features * out_features, 1.f);
  linear.set_weights(weight_vals);

  sftensor input = std::make_shared<ftensor>(1, in_features, in_dims);
  input->Fill(1.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  sftensor output = std::make_shared<ftensor>(1, out_features, in_dims);
  std::vector<sftensor> outputs;
  outputs.push_back(output);

  for (auto _ : state) {
    linear.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Linear)->Args({3, 32, 1000})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({32, 64, 1000})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({64, 128, 1000})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({128, 512, 512})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({128, 1000, 512})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({128, 2048, 512})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({512, 1024, 1000})->Unit(benchmark::kMillisecond);


static void BM_Expression(benchmark::State& state) {
  const int32_t channels = (int32_t)state.range(0);
  const int32_t rows = (int32_t)state.range(1);
  const int32_t cols = (int32_t)state.range(2);

  const std::string& expr = "mul(add(@0,@1),add(@2,@3))";
  Expression expression(expr);
  
  const uint32_t batch = 4;
  std::vector<sftensor> inputs(batch);

  for (uint32_t b = 0; b < batch; ++b) {
    auto& input = inputs.at(b);
    input = std::make_shared<ftensor>(channels, rows, cols);
    input->Fill(1.f * (b + 1));
  }

  std::vector<sftensor> outputs(1);

  for (auto _ : state) {
    expression.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Expression)->Args({3, 320, 320});
BENCHMARK(BM_Expression)->Args({32, 160, 160});
BENCHMARK(BM_Expression)->Args({64, 80, 80});
BENCHMARK(BM_Expression)->Args({128, 40, 40});


static void BM_HardSwish(benchmark::State& state) {

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);  
  for (uint32_t b = 0; b < batch; ++b) {
    auto& input = inputs.at(b);
    input = std::make_shared<ftensor>(channels, rows, cols);
    input->Fill(1.f * b);
  }
  
  std::vector<sftensor> outputs(batch);

  HardSwish hardswish;
  for (auto _ : state) {
    hardswish.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_HardSwish)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSwish)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSwish)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSwish)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);


static void BM_ReLU(benchmark::State& state) {

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);  
  for (uint32_t b = 0; b < batch; ++b) {
    auto& input = inputs.at(b);
    input = std::make_shared<ftensor>(channels, rows, cols);
    input->Fill(1.f * b);
  }
  
  std::vector<sftensor> outputs(batch);

  ReLU relu;
  for (auto _ : state) {
    relu.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_ReLU)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ReLU)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ReLU)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ReLU)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);


static void BM_MaxPooling_k3x3s1x1(benchmark::State& state) {

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  std::vector<sftensor> outputs(1);

  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;

  MaxPooling max_pooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  for (auto _ : state) {
    max_pooling.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_MaxPooling_k3x3s1x1)
    ->Args({3, 320, 320})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MaxPooling_k3x3s1x1)
    ->Args({32, 160, 160})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MaxPooling_k3x3s1x1)
    ->Args({64, 80, 80})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MaxPooling_k3x3s1x1)
    ->Args({128, 40, 40})
    ->Unit(benchmark::kMillisecond);


static void BM_AdaptAvgPooling(benchmark::State& state) {

  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);

  const uint32_t output_h = input_h / 2;
  const uint32_t output_w = input_w / 2;
  
  const uint32_t batch = 3;
  std::vector<sftensor> inputs(batch);
  for (uint32_t b = 0; b < batch; ++b) {
    auto& input = inputs.at(b);
    input = std::make_shared<ftensor>(input_c, input_h, input_w);
    input->Rand();
  }

  std::vector<sftensor> outputs(batch);

  AdaptAvgPooling adapt_avgpooling(output_h, output_w);

  for (auto _ : state) {
    adapt_avgpooling.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_AdaptAvgPooling)
    ->Args({3, 320, 320})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AdaptAvgPooling)
    ->Args({32, 160, 160})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AdaptAvgPooling)
    ->Args({64, 80, 80})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AdaptAvgPooling)
    ->Args({128, 40, 40})
    ->Unit(benchmark::kMillisecond);


static void BM_Flatten(benchmark::State& state) {

  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);
  
  const uint32_t batch = 4;
  std::vector<sftensor> inputs(batch);
  for (uint32_t b = 0; b < batch; ++b) {
    auto& input = inputs.at(b);
    input = std::make_shared<ftensor>(input_c, input_h, input_w);
    input->Rand();
  }

  std::vector<sftensor> outputs(batch);
  
  Flatten flatten(1, 3);
  
  for (auto _ : state) {
    flatten.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Flatten)->Args({1, 224, 224})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Flatten)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Flatten)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Flatten)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Flatten)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);


static void BM_SoftmaxDim1Batch8(benchmark::State& state) {

  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);
  
  const uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);
  for (uint32_t b = 0; b < batch; ++b) {
    auto& input = inputs.at(b);
    input = std::make_shared<ftensor>(input_c, input_h, input_w);
    input->Rand();
  }

  std::vector<sftensor> outputs(batch);
  
  Softmax softmax(1);
  
  for (auto _ : state) {
    softmax.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_SoftmaxDim1Batch8)
    ->Args({1, 224, 224})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SoftmaxDim1Batch8)
    ->Args({8, 128, 128})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SoftmaxDim1Batch8)
    ->Args({3, 320, 320})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SoftmaxDim1Batch8)
    ->Args({32, 160, 160})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SoftmaxDim1Batch8)
    ->Args({64, 80, 80})
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SoftmaxDim1Batch8)
    ->Args({128, 40, 40})
    ->Unit(benchmark::kMillisecond);
