#include "../../src/kernel/details/softmax.hpp"
#include "data/load_data.hpp"
#include "runtime/runtime_graph.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace TinyInfer;

TEST(test_kernel, forward_softmax_dim1) {
  // softmax on dim = 1, raw_shape = (2,3,4)
  RuntimeGraph graph("../../tmp/softmax/softmax_dim1.pnnx.param",
                     "../../tmp/softmax/softmax_dim1.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t val_sz = 24;
  std::vector<float> vals;
  for (uint32_t i = 0; i < val_sz; ++i) {
    vals.push_back(float(i));
  }

  std::vector<sftensor> inputs;
  const uint32_t batch = 4;
  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(2, 3, 4);
    input->Fill(vals, true);
    inputs.push_back(input);
  }

  const auto &outputs = graph.Forward(inputs);

  arma::fmat real =
      CSVDataLoader::LoadData("../../tmp/softmax/softmax_dim1.csv");
  for (const auto &output : outputs) {
    output->Reshape({val_sz}, true);
    for (int i = 0; i < val_sz; ++i) {
      float a = output->index(i);
      float b = real.at(i);
      ASSERT_LE(std::abs(a - b), 1e-5f) << "a: " << a << " b: " << b;
    }
  }
}

TEST(test_kernel, forward_softmax_dim1_minus2) {
  // softmax on dim = -2, raw_shape = (2,3,4)
  RuntimeGraph graph("../../tmp/softmax/softmax_dim1_-2.pnnx.param",
                     "../../tmp/softmax/softmax_dim1_-2.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t val_sz = 24;
  std::vector<float> vals;
  for (uint32_t i = 0; i < val_sz; ++i) {
    vals.push_back(float(i));
  }

  std::vector<sftensor> inputs;
  const uint32_t batch = 4;
  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(2, 3, 4);
    input->Fill(vals, true);
    inputs.push_back(input);
  }

  const auto &outputs = graph.Forward(inputs);

  arma::fmat real =
      CSVDataLoader::LoadData("../../tmp/softmax/softmax_dim1.csv");
  for (const auto &output : outputs) {
    output->Reshape({val_sz}, true);
    for (int i = 0; i < val_sz; ++i) {
      float a = output->index(i);
      float b = real.at(i);
      ASSERT_LE(std::abs(a - b), 1e-5f)
          << "a: " << a << " b: " << b << " i:" << i;
    }
  }
}

TEST(test_kernel, forward_softmax_dim0) {
  Softmax softmax(0);
  for (int k = 0; k < 2; ++k) {
    uint32_t val_sz = 24;
    std::vector<float> vals;
    for (uint32_t i = 0; i < val_sz; ++i) {
      vals.push_back(float(i));
    }

    const uint32_t batch = 4;
    std::vector<sftensor> inputs;

    for (uint32_t b = 0; b < batch; ++b) {
      sftensor input = std::make_shared<ftensor>(2, 3, 4);
      input->Fill(vals, true);
      inputs.push_back(input);
    }

    std::vector<sftensor> outputs(batch);
    softmax.Forward(inputs, outputs);

    arma::fmat real =
        CSVDataLoader::LoadData("../../tmp/softmax/softmax_dim0.csv");
    for (const auto &output : outputs) {
      output->Reshape({val_sz}, true);
      for (int i = 0; i < val_sz; ++i) {
        float a = output->index(i);
        float b = real.at(i);
        ASSERT_LE(std::abs(a - b), 1e-5f);
      }
    }
  }
}

TEST(test_kernel, forward_softmax_dim2) {
  Softmax softmax(2);
  for (int k = 0; k < 2; ++k) {
    uint32_t val_sz = 24;
    std::vector<float> vals;
    for (uint32_t i = 0; i < val_sz; ++i) {
      vals.push_back(float(i));
    }

    const uint32_t batch = 4;
    std::vector<sftensor> inputs;

    for (uint32_t b = 0; b < batch; ++b) {
      sftensor input = std::make_shared<ftensor>(2, 3, 4);
      input->Fill(vals, true);
      inputs.push_back(input);
    }

    std::vector<sftensor> outputs(batch);
    softmax.Forward(inputs, outputs);

    arma::fmat real =
        CSVDataLoader::LoadData("../../tmp/softmax/softmax_dim2.csv");
    for (const auto &output : outputs) {
      output->Reshape({val_sz}, true);
      for (int i = 0; i < val_sz; ++i) {
        float a = output->index(i);
        float b = real.at(i);
        ASSERT_LE(std::abs(a - b), 1e-5f);
      }
    }
  }
}

TEST(test_kernel, forward_softmax_dim1_1) {
  Softmax softmax(0);
  for (int k = 0; k < 2; ++k) {
    uint32_t val_sz = 24;
    std::vector<float> vals;
    for (uint32_t i = 0; i < val_sz; ++i) {
      vals.push_back(float(i));
    }

    const uint32_t batch = 4;
    std::vector<sftensor> inputs;

    for (uint32_t b = 0; b < batch; ++b) {
      sftensor input = std::make_shared<ftensor>(1, 1, 24);
      input->Fill(vals, true);
      inputs.push_back(input);
    }

    std::vector<sftensor> outputs(batch);
    softmax.Forward(inputs, outputs);

    arma::fmat real =
        CSVDataLoader::LoadData("../../tmp/softmax/softmax_dim1_1.csv");
    for (const auto &output : outputs) {
      output->Reshape({val_sz}, true);
      for (int i = 0; i < val_sz; ++i) {
        float a = output->index(i);
        float b = real.at(i);
        ASSERT_LE(std::abs(a - b), 1e-5f);
      }
    }
  }
}

TEST(test_kernel, forward_softmax_dim1_1_m) {
  Softmax softmax(-1);
  for (int k = 0; k < 2; ++k) {
    uint32_t val_sz = 24;
    std::vector<float> vals;
    for (uint32_t i = 0; i < val_sz; ++i) {
      vals.push_back(float(i));
    }

    const uint32_t batch = 4;
    std::vector<sftensor> inputs;

    for (uint32_t b = 0; b < batch; ++b) {
      sftensor input = std::make_shared<ftensor>(1, 1, 24);
      input->Fill(vals, true);
      inputs.push_back(input);
    }

    std::vector<sftensor> outputs(batch);
    softmax.Forward(inputs, outputs);

    arma::fmat real =
        CSVDataLoader::LoadData("../../tmp/softmax/softmax_dim1_1.csv");
    for (const auto &output : outputs) {
      output->Reshape({val_sz}, true);
      for (int i = 0; i < val_sz; ++i) {
        float a = output->index(i);
        float b = real.at(i);
        ASSERT_LE(std::abs(a - b), 1e-5f);
      }
    }
  }
}