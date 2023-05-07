#include "../../src/kernel/details/concat.hpp"
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace TinyInfer;

TEST(test_kernel, concat1) {
  int in_batch = 4;
  int out_batch = 2;
  int in_channels = 6;
  int out_channels = 12;

  std::vector<sftensor> inputs;
  inputs.reserve(in_batch);
  for (int ib = 0; ib < in_batch; ++ib) {
    sftensor input = std::make_shared<ftensor>(in_channels, 32, 32);
    input->Rand();
    inputs.push_back(input);
  }

  std::vector<sftensor> outputs(out_batch);
  Concat concat(1);
  const auto status = concat.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);

  for (uint32_t b = 0; b < out_batch; ++b) {
    ASSERT_EQ(outputs.at(b)->channels(), out_channels);
  }

  for (int ib = 0; ib < in_batch / 2; ++ib) {
    for (int ic = 0; ic < in_channels; ++ic) {
      ASSERT_TRUE(arma::approx_equal(inputs.at(ib)->slice(ic),
                                     outputs.at(ib)->slice(ic), "absdiff",
                                     0.01f));
    }
  }

  for (int ib = in_batch / 2; ib < in_batch; ++ib) {
    for (int ic = in_channels; ic < in_channels * 2; ++ic) {
      ASSERT_TRUE(arma::approx_equal(inputs.at(ib)->slice(ic - in_channels),
                                     outputs.at(ib - out_batch)->slice(ic),
                                     "absdiff", 0.01f));
    }
  }
}

TEST(test_kernel, concat2) {
  int in_batch = 8;
  int out_batch = 4;
  int in_channels = 64;
  int out_channels = 128;

  std::vector<sftensor> inputs;
  inputs.reserve(in_batch);
  for (int ib = 0; ib < in_batch; ++ib) {
    sftensor input = std::make_shared<ftensor>(in_channels, 32, 32);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs(out_batch);
  Concat concat(1);
  const auto status = concat.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);

  for (uint32_t b = 0; b < out_batch; ++b) {
    ASSERT_EQ(outputs.at(b)->channels(), out_channels);
  }

  for (int ib = 0; ib < in_batch / 2; ++ib) {
    for (int ic = 0; ic < in_channels; ++ic) {
      ASSERT_TRUE(arma::approx_equal(inputs.at(ib)->slice(ic),
                                     outputs.at(ib)->slice(ic), "absdiff",
                                     0.01f));
    }
  }

  for (int ib = in_batch / 2; ib < in_batch; ++ib) {
    for (int ic = in_channels; ic < in_channels * 2; ++ic) {
      ASSERT_TRUE(arma::approx_equal(inputs.at(ib)->slice(ic - in_channels),
                                     outputs.at(ib - out_batch)->slice(ic),
                                     "absdiff", 0.01f));
    }
  }
}

TEST(test_kernel, concat3) {
  int in_batch = 3;
  int in_channels = 1;

  int out_batch = 1;
  int out_channels = 3;

  std::vector<sftensor> inputs;
  inputs.reserve(in_batch);
  for (int ib = 0; ib < in_batch; ++ib) {
    sftensor input = std::make_shared<ftensor>(in_channels, 4, 4);
    input->Fill(float(ib) + 1.f);
    inputs.push_back(input);
  }

  std::vector<sftensor> outputs(out_batch);
  Concat concat(1);
  const auto status = concat.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);

  for (uint32_t b = 0; b < out_batch; ++b) {
    ASSERT_EQ(outputs.at(b)->channels(), out_channels);
  }

  for (int ib = 0; ib < in_batch; ++ib) {
    const arma::fmat &in_channel = inputs.at(ib)->slice(0);
    const arma::fmat &out_channel = outputs.at(0)->slice(ib);
    ASSERT_TRUE(arma::approx_equal(in_channel, out_channel, "absdiff", 0.01f));
  }
}

TEST(test_kernel, concat4) {
  int in_batch = 6;
  int in_channels = 1;

  int out_batch = 1;
  int out_channels = 6;

  std::vector<sftensor> inputs;
  inputs.reserve(in_batch);
  for (int ib = 0; ib < in_batch; ++ib) {
    sftensor input = std::make_shared<ftensor>(in_channels, 4, 4);
    input->Fill(float(ib) + 1.f);
    inputs.push_back(input);
  }

  std::vector<sftensor> outputs(out_batch);
  Concat concat(1);
  const auto status = concat.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::InferSuccess);

  for (uint32_t b = 0; b < out_batch; ++b) {
    ASSERT_EQ(outputs.at(b)->channels(), out_channels);
  }

  for (int ib = 0; ib < in_batch; ++ib) {
    const arma::fmat &in_channel = inputs.at(ib)->slice(0);
    const arma::fmat &out_channel = outputs.at(0)->slice(ib);
    ASSERT_TRUE(arma::approx_equal(in_channel, out_channel, "absdiff", 0.01f));
  }
}