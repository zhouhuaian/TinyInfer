#include "layer/abstract/attr_layer.hpp"
#include <glog/logging.h>

namespace TinyInfer {

AttrLayer::AttrLayer(const std::string &name) : Layer(name) {}

void AttrLayer::InitWeights(const uint32_t count, const uint32_t channel,
                            const uint32_t height, const uint32_t width) {
  this->weights_ = std::vector<sftensor>(count);
  // 为每个权重Tensor开辟内存空间
  for (uint32_t i = 0; i < count; ++i) {
    this->weights_.at(i) = std::make_shared<ftensor>(channel, height, width);
  }
}

void AttrLayer::InitBias(const uint32_t count, const uint32_t channel,
                         const uint32_t height, const uint32_t width) {
  this->bias_ = std::vector<sftensor>(count);
  //为每个偏置Tensor开辟内存空间
  for (uint32_t i = 0; i < count; ++i) {
    this->bias_.at(i) = std::make_shared<ftensor>(channel, height, width);
  }
}

void AttrLayer::set_weights(const std::vector<sftensor> &weights) {
  if (!this->weights_.empty()) {
    CHECK(this->weights_.size() == weights.size());

    for (uint32_t i = 0; i < weights.size(); ++i) {
      CHECK(this->weights_.at(i) != nullptr);
      CHECK(this->weights_.at(i)->channels() == weights.at(i)->channels());
      CHECK(this->weights_.at(i)->rows() == weights.at(i)->rows());
      CHECK(this->weights_.at(i)->cols() == weights.at(i)->cols());
    }
  }

  this->weights_ = weights;
}

void AttrLayer::set_weights(const std::vector<float> &weights) {
  const uint32_t elem_ct = weights.size(); // weights中元素总数

  const uint32_t weight_ct = this->weights_.size(); // 权重总数
  uint32_t val_ct = 0; // 所有权重中的元素总数
  for (uint32_t i = 0; i < weight_ct; ++i) {
    val_ct += this->weights_.at(i)->size();
  }

  CHECK_EQ(val_ct, elem_ct);
  CHECK_EQ(elem_ct % weight_ct, 0);

  const uint32_t blob_ct = elem_ct / weight_ct; // 单个权重中的元素总数
  for (uint32_t i = 0; i < weight_ct; ++i) {
    const uint32_t start = i * blob_ct;
    const uint32_t end = start + blob_ct;
    const auto &sub_vals =
        std::vector<float>{weights.begin() + start, weights.begin() + end};
    this->weights_.at(i)->Fill(sub_vals, true);
  }
}

void AttrLayer::set_bias(const std::vector<sftensor> &bias) {
  if (!this->bias_.empty()) {
    CHECK(this->bias_.size() == bias.size());

    for (uint32_t i = 0; i < bias.size(); ++i) {
      CHECK(this->bias_.at(i) != nullptr);
      CHECK(this->bias_.at(i)->channels() == bias.at(i)->channels());
      CHECK(this->bias_.at(i)->rows() == bias.at(i)->rows());
      CHECK(this->bias_.at(i)->cols() == bias.at(i)->cols());
    }
  }

  this->bias_ = bias;
}

void AttrLayer::set_bias(const std::vector<float> &bias) {
  const uint32_t elem_size = bias.size();

  const uint32_t bias_size = this->bias_.size(); // 偏置总数
  uint32_t val_size = 0; // 所有偏置中的元素总数
  for (uint32_t i = 0; i < bias_size; ++i) {
    val_size += this->bias_.at(i)->size();
  }

  CHECK_EQ(val_size, elem_size);
  CHECK_EQ(elem_size % bias_size, 0);

  const uint32_t blob_size = elem_size / bias_size; // 单个偏置中的元素总数
  for (uint32_t i = 0; i < bias_size; ++i) {
    const uint32_t start = i * blob_size;
    const uint32_t end = start + blob_size;
    const auto &sub_vals =
        std::vector<float>{bias.begin() + start, bias.begin() + end};
    this->bias_.at(i)->Fill(sub_vals, true);
  }
}

const std::vector<sftensor> &AttrLayer::weights() const {
  return this->weights_;
}

const std::vector<sftensor> &AttrLayer::bias() const { return this->bias_; }

} // namespace TinyInfer
