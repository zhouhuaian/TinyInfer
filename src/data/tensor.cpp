#include "data/tensor.hpp"
#include <glog/logging.h>
#include <memory>

namespace TinyInfer {

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  CHECK(channels >= 1 && rows >= 1 && cols >= 1);

  data_ = arma::fcube(rows, cols, channels);
  if (channels == 1 && rows == 1) { // (1,1,cols)->(cols)
    this->raw_shape_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) { // (1,rows,cols)->(rows,cols)
    this->raw_shape_ = std::vector<uint32_t>{rows, cols};
  } else { // (channels,rows,cols)
    this->raw_shape_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

Tensor<float>::Tensor(const std::vector<uint32_t> &shape) {
  CHECK(shape.size() == 3);

  uint32_t channels = shape.at(0);
  uint32_t rows = shape.at(1);
  uint32_t cols = shape.at(2);

  data_ = arma::fcube(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shape_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shape_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shape_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

Tensor<float>::Tensor(const Tensor &tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shape_ = tensor.raw_shape_;
  }
}

Tensor<float>::Tensor(ftensor &&tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shape_ = std::move(tensor.raw_shape_);
  }
}

ftensor &Tensor<float>::operator=(const Tensor &tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shape_ = tensor.raw_shape_;
  }
  return *this;
}

ftensor &Tensor<float>::operator=(ftensor &&tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shape_ = std::move(tensor.raw_shape_);
  }
  return *this;
}

uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

void Tensor<float>::set_data(const arma::fcube &data) {
  CHECK(data.n_rows == this->data_.n_rows)
      << data.n_rows << " != " << this->data_.n_rows;
  CHECK(data.n_cols == this->data_.n_cols)
      << data.n_cols << " != " << this->data_.n_cols;
  CHECK(data.n_slices == this->data_.n_slices)
      << data.n_slices << " != " << this->data_.n_slices;
  this->data_ = data;
}

bool Tensor<float>::empty() const { return this->data_.empty(); }

std::vector<uint32_t> Tensor<float>::shape() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

const std::vector<uint32_t> &Tensor<float>::raw_shape() const {
  CHECK(!this->raw_shape_.empty());
  return this->raw_shape_;
}

float &Tensor<float>::index(uint32_t offset) {
  CHECK(offset < this->data_.size()) << "Tensor capacity is not enough!";
  return this->data_.at(offset);
}

float Tensor<float>::index(uint32_t offset) const {
  CHECK(offset < this->data_.size()) << "Tensor capacity is not enough!";
  return this->data_.at(offset);
}

arma::fcube &Tensor<float>::data() { return this->data_; }

const arma::fcube &Tensor<float>::data() const { return this->data_; }

arma::fmat &Tensor<float>::slice(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::fmat &Tensor<float>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

float &Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

void Tensor<float>::Pad(const std::vector<uint32_t> &pads, float pad_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);

  uint32_t pad_rows1 = pads.at(0); // up
  uint32_t pad_rows2 = pads.at(1); // bottom
  uint32_t pad_cols1 = pads.at(2); // left
  uint32_t pad_cols2 = pads.at(3); // right

  arma::fcube new_data(this->data_.n_rows + pad_rows1 + pad_rows2,
                       this->data_.n_cols + pad_cols1 + pad_cols2,
                       this->data_.n_slices);
  new_data.fill(pad_value);

  new_data.subcube(pad_rows1, pad_cols1, 0, new_data.n_rows - pad_rows2 - 1,
                   new_data.n_cols - pad_cols2 - 1, new_data.n_slices - 1) =
      this->data_;
  this->data_ = std::move(new_data);
}

void Tensor<float>::Fill(float value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

void Tensor<float>::Fill(const std::vector<float> &values, bool row_major) {
  CHECK(!this->data_.empty());
  CHECK_EQ(values.size(), this->data_.size());

  // 行主序填充
  if (row_major) {
    const uint32_t rows = this->rows();
    const uint32_t cols = this->cols();
    const uint32_t planes = rows * cols;
    const uint32_t channels = this->data_.n_slices;

    for (uint32_t i = 0; i < channels; ++i) {
      auto &channel_data = this->data_.slice(i);
      const arma::fmat &channel_data_t =
          arma::fmat(values.data() + i * planes, this->cols(), this->rows());
      channel_data = channel_data_t.t();
    }
  }
  // 列主序填充
  else {
    std::copy(values.begin(), values.end(), this->data_.memptr());
  }
}

void Tensor<float>::Ones() {
  CHECK(!this->data_.empty());
  this->Fill(1.f);
}

void Tensor<float>::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();
}

void Tensor<float>::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i << "\n";
    LOG(INFO) << this->data_.slice(i);
  }
}

void Tensor<float>::Reshape(const std::vector<uint32_t> &shape,
                            bool row_major) {
  CHECK(!this->data_.empty());
  CHECK(!shape.empty() && shape.size() <= 3);

  const uint32_t origin_size = this->size();
  uint32_t current_size = 1;
  for (uint32_t s : shape) {
    current_size *= s;
  }
  CHECK_EQ(current_size, origin_size);

  // 以行主序重排张量中的元素
  if (row_major) {
    std::vector<uint32_t> target_shapes; // (channel, row, col)
    if (shape.size() == 3) {
      target_shapes = {shape.at(0), shape.at(1), shape.at(2)};
      this->raw_shape_ = {shape.at(0), shape.at(1), shape.at(2)};
    } else if (shape.size() == 2) {
      target_shapes = {1, shape.at(0), shape.at(1)};
      this->raw_shape_ = {shape.at(0), shape.at(1)};
    } else {
      target_shapes = {1, shape.at(0), 1};
      this->raw_shape_ = {shape.at(0)};
    }
    this->ReView(target_shapes);
  }
  // 以列主序重排张量中的元素
  else {
    if (shape.size() == 3) {
      this->data_.reshape(shape.at(1), shape.at(2), shape.at(0));
      this->raw_shape_ = {shape.at(0), shape.at(1), shape.at(2)};
    } else if (shape.size() == 2) {
      this->data_.reshape(shape.at(0), shape.at(1), 1);
      this->raw_shape_ = {shape.at(0), shape.at(1)};
    } else {
      this->data_.reshape(shape.at(0), 1, 1);
      this->raw_shape_ = {shape.at(0)};
    }
  }
}

void Tensor<float>::Flatten(bool row_major) {
  CHECK(!this->data_.empty());
  const uint32_t size = this->data_.size();
  this->Reshape({size}, row_major);
}

std::vector<float> Tensor<float>::values(bool row_major) {
  CHECK_EQ(this->data_.empty(), false);

  std::vector<float> values(this->data_.size());
  // 行主序
  if (row_major) {
    uint32_t index = 0;
    for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
      const arma::fmat &channel = this->data_.slice(c).t();
      std::copy(channel.begin(), channel.end(), values.begin() + index);
      index += channel.size();
    }
    CHECK_EQ(index, values.size());
  }
  // 列主序
  else {
    std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
              values.begin());
  }

  return values;
}

void Tensor<float>::Transform(const std::function<float(float)> &filter) {
  CHECK(!this->data_.empty());
  this->data_.transform(filter);
}

sftensor Tensor<float>::Clone() { return std::make_shared<Tensor>(*this); }

const float *Tensor<float>::raw_ptr() const {
  CHECK(!this->data_.empty());
  return this->data_.memptr();
}

void Tensor<float>::ReView(const std::vector<uint32_t> &shape) {
  CHECK(!this->data_.empty());

  const uint32_t target_channels = shape.at(0);
  const uint32_t target_rows = shape.at(1);
  const uint32_t target_cols = shape.at(2);
  CHECK_EQ(this->data_.size(), target_channels * target_cols * target_rows);

  arma::fcube new_data(target_rows, target_cols, target_channels);
  const uint32_t plane_size = target_rows * target_cols;
  for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
    const arma::fmat &channel = this->data_.slice(c);
    for (uint32_t c_ = 0; c_ < this->data_.n_cols; ++c_) {
      const float *col_ptr = channel.colptr(c_);
      for (uint32_t r = 0; r < this->data_.n_rows; ++r) {
        // 计算元素在原张量中以行主序的偏移量
        const uint32_t pos_index =
            c * data_.n_rows * data_.n_cols + r * data_.n_cols + c_;
        // 计算重排后该元素的位置
        const uint32_t ch = pos_index / plane_size;
        const uint32_t row = (pos_index - ch * plane_size) / target_cols;
        const uint32_t col = (pos_index - ch * plane_size - row * target_cols);
        CHECK(ch < new_data.n_slices && col < new_data.n_cols &&
              row < new_data.n_rows);
        new_data.at(row, col, ch) = *(col_ptr + r);
      }
    }
  }
  this->data_ = std::move(new_data);
}

bool IsSame(const sftensor &in1, const sftensor &in2) {
  CHECK(in1 != nullptr && in2 != nullptr);

  if (in1->shape() != in2->shape()) {
    return false;
  }
  bool is_same = arma::approx_equal(in1->data(), in2->data(), "absdiff", 1e-5);
  return is_same;
}

sftensor ElemAdd(const sftensor &in1, const sftensor &in2) {
  CHECK(in1 != nullptr && in2 != nullptr);

  if (in1->shape() == in2->shape()) {
    sftensor output_tensor = Create(in1->shape());
    output_tensor->set_data(in1->data() + in2->data());
    return output_tensor;
  } else {
    // Broadcast
    CHECK(in1->channels() == in2->channels())
        << "Tensors shape are not adapting";
    const auto &[input1, input2] = Broadcast(in1, in2);
    CHECK(input1->shape() == input2->shape());
    sftensor output_tensor = Create(input1->shape());
    output_tensor->set_data(input1->data() + input2->data());
    return output_tensor;
  }
}

void ElemAdd(const sftensor &in1, const sftensor &in2, const sftensor &out) {
  out->set_data(ElemAdd(in1, in2)->data());
}

sftensor ElemMul(const sftensor &in1, const sftensor &in2) {
  CHECK(in1 != nullptr && in2 != nullptr);

  if (in1->shape() == in2->shape()) {
    sftensor output_tensor = Create(in1->shape());
    output_tensor->set_data(in1->data() % in2->data());
    return output_tensor;
  } else {
    // Broadcast
    CHECK(in1->channels() == in2->channels())
        << "Tensors shape are not adapting";
    const auto &[input1, input2] = Broadcast(in1, in2);
    CHECK(input1->shape() == input2->shape());
    sftensor output_tensor = Create(input1->shape());
    output_tensor->set_data(input1->data() % input2->data());
    return output_tensor;
  }
}

void ElemMul(const sftensor &in1, const sftensor &in2, const sftensor &out) {
  out->set_data(ElemMul(in1, in2)->data());
}

sftensor Create(uint32_t channels, uint32_t rows, uint32_t cols) {
  CHECK(channels >= 1 && rows >= 1 && cols >= 1);
  return std::make_shared<ftensor>(channels, rows, cols);
}

sftensor Create(const std::vector<uint32_t> &shape) {
  CHECK(shape.size() == 3);
  return Create(shape.at(0), shape.at(1), shape.at(2));
}

std::tuple<sftensor, sftensor> Broadcast(const sftensor &in1,
                                         const sftensor &in2) {
  CHECK(in1 != nullptr && in2 != nullptr);
  if (in1->shape() == in2->shape()) {
    return {in1, in2};
  } else {
    CHECK(in1->channels() == in2->channels());
    if (in2->rows() == 1 && in2->cols() == 1) {
      sftensor new_tensor = Create(in2->channels(), in1->rows(), in1->cols());
      for (uint32_t c = 0; c < in2->channels(); ++c) {
        new_tensor->slice(c).fill(in2->index(c));
      }
      return {in1, new_tensor};
    } else if (in1->rows() == 1 && in1->cols() == 1) {
      sftensor new_tensor = Create(in1->channels(), in2->rows(), in2->cols());
      for (uint32_t c = 0; c < in1->channels(); ++c) {
        new_tensor->slice(c).fill(in1->index(c));
      }
      return {new_tensor, in2};
    } else {
      LOG(FATAL) << "Broadcast shape is not adapting!";
      return {in1, in2};
    }
  }
}

sftensor Pad(const sftensor &tensor, const std::vector<uint32_t> &pads,
             float pad_value) {
  CHECK(tensor != nullptr && !tensor->empty());
  CHECK(pads.size() == 4);

  uint32_t pad_rows1 = pads.at(0); // up
  uint32_t pad_rows2 = pads.at(1); // bottom
  uint32_t pad_cols1 = pads.at(2); // left
  uint32_t pad_cols2 = pads.at(3); // right

  sftensor output = std::make_shared<ftensor>(
      tensor->channels(), tensor->rows() + pad_rows1 + pad_rows2,
      tensor->cols() + pad_cols1 + pad_cols2);

  if (pad_value != 0.f)
    output->Fill(pad_value);

  const uint32_t channels = tensor->channels();
  for (uint32_t channel = 0; channel < channels; ++channel) {
    const arma::fmat &in_channel = tensor->slice(channel);
    arma::fmat &output_channel = output->slice(channel);
    const uint32_t in_channel_width = in_channel.n_cols;
    const uint32_t in_channel_height = in_channel.n_rows;

    for (uint32_t w = 0; w < in_channel_width; ++w) {
      float *output_channel_ptr =
          const_cast<float *>(output_channel.colptr(w + pad_cols1));
      const float *in_channel_ptr = in_channel.colptr(w);
      for (uint32_t h = 0; h < in_channel_height; ++h) {
        const float value = *(in_channel_ptr + h);
        *(output_channel_ptr + h + pad_rows1) = value;
      }
    }
  }
  return output;
}

} // namespace TinyInfer
