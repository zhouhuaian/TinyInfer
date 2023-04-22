#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../../src/layer/details/maxpooling.hpp"

using namespace TinyInfer;

void MaxPoolingFunc(const std::vector<sftensor>& inputs, std::vector<sftensor>& outputs,
                    int stride_w, int stride_h, int kernel_h, int kernel_w) {
  const uint32_t batch = inputs.size();
  for (uint32_t b = 0; b < batch; ++b) {

    const sftensor& input = inputs.at(b);
    const uint32_t input_c = input->channels();
    const uint32_t input_h = input->rows();
    const uint32_t input_w = input->cols();
    
    const uint32_t output_c = input_c;
    const uint32_t output_h = uint32_t(std::floor((input_h - kernel_h) / stride_h + 1));
    const uint32_t output_w = uint32_t(std::floor((input_w - kernel_w) / stride_w + 1));
    CHECK(output_w > 0 && output_h > 0);

    sftensor output = std::make_shared<ftensor>(output_c, output_h, output_w);
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat& in_channel = input->slice(ic);
      arma::fmat& out_channel = output->slice(ic);
      for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
          const arma::fmat& window = in_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
          out_channel.at(int(r / stride_h), int(c / stride_w)) = window.max();
        }
      }
    }
    outputs.push_back(output);
  }
}

TEST(test_layer, forward_max_pooling_s11_k33) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;

  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s11_k33_p1) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 226;
  const uint32_t input_w = 226;
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;

  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s22_k33) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;

  const uint32_t stride_h = 2;
  const uint32_t stride_w = 2;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s33_k33) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;

  const uint32_t stride_h = 3;
  const uint32_t stride_w = 3;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s11_k55) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;

  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s22_k55) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;

  const uint32_t stride_h = 2;
  const uint32_t stride_w = 2;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s33_k55) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;

  const uint32_t stride_h = 3;
  const uint32_t stride_w = 3;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s55_k55) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;

  const uint32_t stride_h = 5;
  const uint32_t stride_w = 5;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s11_k77) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 7;
  const uint32_t kernel_w = 7;

  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s22_k77) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 7;
  const uint32_t kernel_w = 7;

  const uint32_t stride_h = 2;
  const uint32_t stride_w = 2;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s33_k77) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 7;
  const uint32_t kernel_w = 7;

  const uint32_t stride_h = 3;
  const uint32_t stride_w = 3;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s55_k77) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 7;
  const uint32_t kernel_w = 7;

  const uint32_t stride_h = 5;
  const uint32_t stride_w = 5;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s77_k77) {
  std::vector<sftensor> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 7;
  const uint32_t kernel_w = 7;

  const uint32_t stride_h = 7;
  const uint32_t stride_w = 7;
  const uint32_t batch = 3;

  for (uint32_t b = 0; b < batch; ++b) {
    sftensor input = std::make_shared<ftensor>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs1;
  MaxPoolingFunc(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), batch);

  std::vector<sftensor> outputs2(batch);
  MaxPooling maxpooling(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  maxpooling.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), batch);

  for (uint32_t b = 0; b < batch; ++b) {
    const auto& output1 = outputs1.at(b);
    const auto& output2 = outputs2.at(b);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}