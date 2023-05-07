#include "../../src/kernel/details/convolution.hpp"
#include "data/tensor.hpp"
#include <cstdint>
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace TinyInfer;

InferStatus ConvolutionFunc(const std::vector<sftensor> &inputs,
                            std::vector<sftensor> &outputs,
                            const uint32_t stride_h_, const uint32_t stride_w_,
                            const std::vector<sftensor> &weights_) {
  if (inputs.empty()) {
    LOG(ERROR) << "Input tensor array empty";
    return InferStatus::InferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "Input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  CHECK(stride_h_ > 0 && stride_w_ > 0) << "Stride must greater than 0";

  CHECK(!weights_.empty()) << "Weight count must greater than 0";
  const uint32_t kernel_ct = weights_.size();
  CHECK(weights_.at(0) != nullptr && !weights_.at(0)->empty())
      << "Weight empty";
  const uint32_t kernel_h = weights_.at(0)->rows();
  const uint32_t kernel_w = weights_.at(0)->cols();

  const uint32_t batch = inputs.size();

  for (uint32_t b = 0; b < batch; ++b) {
    const auto &input = inputs.at(b);
    auto &output = outputs.at(b);

    CHECK(input != nullptr && !input->empty()) << b << " input tensor empty";

    const uint32_t input_c = input->channels();
    const uint32_t input_h = input->rows();
    const uint32_t input_w = input->cols();

    for (uint32_t k = 0; k < kernel_ct; ++k) {
      const auto &kernel = weights_.at(k);
      CHECK(kernel->channels() == input_c && kernel->rows() == kernel_h &&
            kernel->cols() == kernel_w)
          << k << "kernel shape error";

      uint32_t output_h =
          uint32_t(std::floor((input_h - kernel_h) / stride_h_ + 1));
      uint32_t output_w =
          uint32_t(std::floor((input_w - kernel_w) / stride_w_ + 1));

      CHECK(output_h > 0 && output_w > 0) << "Output shape error";

      if (!output) {
        output = std::make_shared<ftensor>(kernel_ct, output_h, output_w);
        output->Fill(0.f);
      }

      arma::fmat &out_channel = output->slice(k);
      for (uint32_t ic = 0; ic < input_c; ++ic) {
        const arma::fmat &in_channels = input->slice(ic);
        const arma::fmat &kernel_channel = kernel->slice(ic);
        // 从左至右，从上至下滑动
        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w_) {
          for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h_) {
            const arma::fmat &window =
                in_channels.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
            const float sum_val = arma::accu(window % kernel_channel);
            out_channel.at(int(r / stride_h_), int(c / stride_w_)) += sum_val;
          }
        }
      }
    }
    CHECK(!output->empty()) << b << " output tensor empty";
  }

  return InferStatus::InferSuccess;
}

TEST(test_kernel, conv3x3x32_stride1x1_padding0) {
  const uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);
  std::vector<sftensor> outputs1(batch);
  std::vector<sftensor> outputs2(batch);

  const uint32_t in_channels = 32;
  for (uint32_t b = 0; b < batch; ++b) {
    inputs.at(b) = std::make_shared<ftensor>(in_channels, 8, 8);
    inputs.at(b)->Rand();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_ct = 8;
  std::vector<sftensor> weights(kernel_ct);
  for (uint32_t k = 0; k < kernel_ct; ++k) {
    weights.at(k) = std::make_shared<ftensor>(in_channels, kernel_h, kernel_w);
    weights.at(k)->Rand();
  }
  ConvolutionFunc(inputs, outputs1, stride_h, stride_w, weights);
  Convolution conv(kernel_ct, in_channels, kernel_h, kernel_w, 0, 0, stride_h,
                   stride_w, 1, false);
  conv.set_weights(weights);
  conv.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t b = 0; b < batch; ++b) {
    ASSERT_EQ(outputs1.at(b)->size(), outputs2.at(b)->size());
    const uint32_t out_size = outputs1.at(b)->size();
    for (uint32_t i = 0; i < out_size; ++i) {
      ASSERT_LE(std::abs(outputs1.at(b)->index(i) - outputs2.at(b)->index(i)),
                1e-4);
    }
  }
}

TEST(test_kernel, conv3x3x32_stride1x1_padding2) {
  const uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);
  std::vector<sftensor> outputs1(batch);
  std::vector<sftensor> outputs2(batch);

  const uint32_t in_channels = 32;
  for (uint32_t b = 0; b < batch; ++b) {
    inputs.at(b) = std::make_shared<ftensor>(in_channels, 226, 226);
    inputs.at(b)->Rand();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_ct = 8;
  std::vector<sftensor> weights(kernel_ct);
  for (uint32_t k = 0; k < kernel_ct; ++k) {
    weights.at(k) = std::make_shared<ftensor>(in_channels, kernel_h, kernel_w);
    weights.at(k)->Rand();
  }
  ConvolutionFunc(inputs, outputs1, stride_h, stride_w, weights);
  Convolution conv(kernel_ct, in_channels, kernel_h, kernel_w, 0, 0, stride_h,
                   stride_w, 1, false);
  conv.set_weights(weights);
  conv.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t b = 0; b < batch; ++b) {
    ASSERT_EQ(outputs1.at(b)->size(), outputs2.at(b)->size());
    const uint32_t out_size = outputs1.at(b)->size();
    for (uint32_t i = 0; i < out_size; ++i) {
      ASSERT_LE(std::abs(outputs1.at(b)->index(i) - outputs2.at(b)->index(i)),
                1e-4);
    }
  }
}

TEST(test_kernel, conv3x3x32_stride2x2_padding2) {
  const uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);
  std::vector<sftensor> outputs1(batch);
  std::vector<sftensor> outputs2(batch);

  const uint32_t in_channels = 32;
  for (uint32_t b = 0; b < batch; ++b) {
    inputs.at(b) = std::make_shared<ftensor>(in_channels, 226, 226);
    inputs.at(b)->Rand();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 2;
  const uint32_t stride_w = 2;
  const uint32_t kernel_ct = 8;
  std::vector<sftensor> weights(kernel_ct);
  for (uint32_t k = 0; k < kernel_ct; ++k) {
    weights.at(k) = std::make_shared<ftensor>(in_channels, kernel_h, kernel_w);
    weights.at(k)->Rand();
  }
  ConvolutionFunc(inputs, outputs1, stride_h, stride_w, weights);
  Convolution conv(kernel_ct, in_channels, kernel_h, kernel_w, 0, 0, stride_h,
                   stride_w, 1, false);
  conv.set_weights(weights);
  conv.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t b = 0; b < batch; ++b) {
    ASSERT_EQ(outputs1.at(b)->size(), outputs2.at(b)->size());
    const uint32_t out_size = outputs1.at(b)->size();
    for (uint32_t i = 0; i < out_size; ++i) {
      ASSERT_LE(std::abs(outputs1.at(b)->index(i) - outputs2.at(b)->index(i)),
                1e-4);
    }
  }
}

TEST(test_kernel, conv3x3x32_stride5x5_padding2) {
  const uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);
  std::vector<sftensor> outputs1(batch);
  std::vector<sftensor> outputs2(batch);

  const uint32_t in_channels = 32;
  for (uint32_t b = 0; b < batch; ++b) {
    inputs.at(b) = std::make_shared<ftensor>(in_channels, 226, 226);
    inputs.at(b)->Rand();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 5;
  const uint32_t stride_w = 5;
  const uint32_t kernel_ct = 8;
  std::vector<sftensor> weights(kernel_ct);
  for (uint32_t k = 0; k < kernel_ct; ++k) {
    weights.at(k) = std::make_shared<ftensor>(in_channels, kernel_h, kernel_w);
    weights.at(k)->Rand();
  }
  ConvolutionFunc(inputs, outputs1, stride_h, stride_w, weights);
  Convolution conv(kernel_ct, in_channels, kernel_h, kernel_w, 0, 0, stride_h,
                   stride_w, 1, false);
  conv.set_weights(weights);
  conv.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t b = 0; b < batch; ++b) {
    ASSERT_EQ(outputs1.at(b)->size(), outputs2.at(b)->size());
    const uint32_t out_size = outputs1.at(b)->size();
    for (uint32_t i = 0; i < out_size; ++i) {
      ASSERT_LE(std::abs(outputs1.at(b)->index(i) - outputs2.at(b)->index(i)),
                1e-4);
    }
  }
}

TEST(test_kernel, conv5x5x32_stride5x5_padding2) {
  const uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);
  std::vector<sftensor> outputs1(batch);
  std::vector<sftensor> outputs2(batch);

  const uint32_t in_channels = 32;
  for (uint32_t b = 0; b < batch; ++b) {
    inputs.at(b) = std::make_shared<ftensor>(in_channels, 226, 226);
    inputs.at(b)->Rand();
  }
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;
  const uint32_t stride_h = 5;
  const uint32_t stride_w = 5;
  const uint32_t kernel_ct = 8;
  std::vector<sftensor> weights(kernel_ct);
  for (uint32_t k = 0; k < kernel_ct; ++k) {
    weights.at(k) = std::make_shared<ftensor>(in_channels, kernel_h, kernel_w);
    weights.at(k)->Rand();
  }
  ConvolutionFunc(inputs, outputs1, stride_h, stride_w, weights);
  Convolution conv(kernel_ct, in_channels, kernel_h, kernel_w, 0, 0, stride_h,
                   stride_w, 1, false);
  conv.set_weights(weights);
  conv.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t b = 0; b < batch; ++b) {
    ASSERT_EQ(outputs1.at(b)->size(), outputs2.at(b)->size());
    const uint32_t out_size = outputs1.at(b)->size();
    for (uint32_t i = 0; i < out_size; ++i) {
#ifdef _MSC_VER
      ASSERT_LE(std::abs(outputs1.at(b)->index(i) - outputs2.at(b)->index(i)),
                1e-3);
#else
      ASSERT_LE(std::abs(outputs1.at(b)->index(i) - outputs2.at(b)->index(i)),
                1e-4);
#endif
    }
  }
}

TEST(test_kernel, conv5x5x32_stride7x7_padding2) {
  const uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);
  std::vector<sftensor> outputs1(batch);
  std::vector<sftensor> outputs2(batch);

  const uint32_t in_channels = 32;
  for (uint32_t b = 0; b < batch; ++b) {
    inputs.at(b) = std::make_shared<ftensor>(in_channels, 226, 226);
    inputs.at(b)->Rand();
  }
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;
  const uint32_t stride_h = 7;
  const uint32_t stride_w = 7;
  const uint32_t kernel_ct = 8;
  std::vector<sftensor> weights(kernel_ct);
  for (uint32_t k = 0; k < kernel_ct; ++k) {
    weights.at(k) = std::make_shared<ftensor>(in_channels, kernel_h, kernel_w);
    weights.at(k)->Rand();
  }
  ConvolutionFunc(inputs, outputs1, stride_h, stride_w, weights);
  Convolution conv(kernel_ct, in_channels, kernel_h, kernel_w, 0, 0, stride_h,
                   stride_w, 1, false);
  conv.set_weights(weights);
  conv.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t b = 0; b < batch; ++b) {
    ASSERT_EQ(outputs1.at(b)->size(), outputs2.at(b)->size());
    const uint32_t out_size = outputs1.at(b)->size();
    for (uint32_t i = 0; i < out_size; ++i) {
#ifdef _MSC_VER
      ASSERT_LE(std::abs(outputs1.at(b)->index(i) - outputs2.at(b)->index(i)),
                1e-3);
#else
      ASSERT_LE(std::abs(outputs1.at(b)->index(i) - outputs2.at(b)->index(i)),
                1e-4);
#endif
    }
  }
}

TEST(test_kernel, conv13x13x32_stride7x7_padding2) {
  const uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);
  std::vector<sftensor> outputs1(batch);
  std::vector<sftensor> outputs2(batch);

  const uint32_t in_channels = 32;
  for (uint32_t b = 0; b < batch; ++b) {
    inputs.at(b) = std::make_shared<ftensor>(in_channels, 226, 226);
    inputs.at(b)->Rand();
  }
  const uint32_t kernel_h = 13;
  const uint32_t kernel_w = 13;
  const uint32_t stride_h = 7;
  const uint32_t stride_w = 7;
  const uint32_t kernel_ct = 8;
  std::vector<sftensor> weights(kernel_ct);
  for (uint32_t k = 0; k < kernel_ct; ++k) {
    weights.at(k) = std::make_shared<ftensor>(in_channels, kernel_h, kernel_w);
    weights.at(k)->Rand();
  }
  ConvolutionFunc(inputs, outputs1, stride_h, stride_w, weights);
  Convolution conv(kernel_ct, in_channels, kernel_h, kernel_w, 0, 0, stride_h,
                   stride_w, 1, false);
  conv.set_weights(weights);
  conv.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t b = 0; b < batch; ++b) {
    ASSERT_EQ(outputs1.at(b)->size(), outputs2.at(b)->size());
    const uint32_t out_size = outputs1.at(b)->size();
    for (uint32_t i = 0; i < out_size; ++i) {
      ASSERT_LE(std::abs(outputs1.at(b)->index(i) - outputs2.at(b)->index(i)),
                1e-3);
    }
  }
}

TEST(test_kernel, conv13x13x31_stride19x19_padding2) {
  const uint32_t batch = 8;
  std::vector<sftensor> inputs(batch);
  std::vector<sftensor> outputs1(batch);
  std::vector<sftensor> outputs2(batch);

  const uint32_t in_channels = 31;
  for (uint32_t b = 0; b < batch; ++b) {
    inputs.at(b) = std::make_shared<ftensor>(in_channels, 340, 340);
    inputs.at(b)->Rand();
  }
  const uint32_t kernel_h = 13;
  const uint32_t kernel_w = 13;
  const uint32_t stride_h = 19;
  const uint32_t stride_w = 19;
  const uint32_t kernel_ct = 8;
  std::vector<sftensor> weights(kernel_ct);
  for (uint32_t k = 0; k < kernel_ct; ++k) {
    weights.at(k) = std::make_shared<ftensor>(in_channels, kernel_h, kernel_w);
    weights.at(k)->Rand();
  }
  ConvolutionFunc(inputs, outputs1, stride_h, stride_w, weights);
  Convolution conv(kernel_ct, in_channels, kernel_h, kernel_w, 0, 0, stride_h,
                   stride_w, 1, false);
  conv.set_weights(weights);
  conv.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t b = 0; b < batch; ++b) {
    ASSERT_EQ(outputs1.at(b)->size(), outputs2.at(b)->size());
    const uint32_t out_size = outputs1.at(b)->size();
    for (uint32_t i = 0; i < out_size; ++i) {
      ASSERT_LE(std::abs(outputs1.at(b)->index(i) - outputs2.at(b)->index(i)),
                1e-3);
    }
  }
}