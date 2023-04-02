#include <glog/logging.h>
#include <gtest/gtest.h>
#include "data/tensor.hpp"

using namespace TinyInfer;

TEST(test_tensor, tensor_init1) {
  ftensor f1(3, 224, 224);
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);
  ASSERT_EQ(f1.size(), 224 * 224 * 3);
}

TEST(test_tensor, tensor_init2) {
  ftensor f1(std::vector<uint32_t>{3, 224, 224});
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);
  ASSERT_EQ(f1.size(), 224 * 224 * 3);
}

TEST(test_tensor, copy_construct1) {
  ftensor f1(3, 224, 224);
  f1.Rand();
  ftensor f2(f1);
  ASSERT_EQ(f2.channels(), 3);
  ASSERT_EQ(f2.rows(), 224);
  ASSERT_EQ(f2.cols(), 224);

  ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, copy_construct2) {
  ftensor f1(3, 2, 1);
  ftensor f2(3, 224, 224);
  f2.Rand();
  f1 = f2;
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);

  ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, copy_construct3) {
  ftensor f1(3, 2, 1);
  ftensor f2(std::vector<uint32_t>{3, 224, 224});
  f2.Rand();
  f1 = f2;
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);

  ASSERT_TRUE(arma::approx_equal(f2.data(), f1.data(), "absdiff", 1e-4));
}

TEST(test_tensor, move_construct1) {
  ftensor f1(3, 2, 1);
  ftensor f2(3, 224, 224);
  f1 = std::move(f2);
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);

  ASSERT_EQ(f2.data().memptr(), nullptr);
}

TEST(test_tensor, move_construct2) {
  ftensor f2(3, 224, 224);
  ftensor f1(std::move(f2));
  ASSERT_EQ(f1.channels(), 3);
  ASSERT_EQ(f1.rows(), 224);
  ASSERT_EQ(f1.cols(), 224);

  ASSERT_EQ(f2.data().memptr(), nullptr);
}

TEST(test_tensor, set_data) {
  ftensor f2(3, 224, 224);
  arma::fcube cube1(224, 224, 3);
  cube1.randn();
  f2.set_data(cube1);

  ASSERT_TRUE(arma::approx_equal(f2.data(), cube1, "absdiff", 1e-4));
}

TEST(test_tensor, data) {
  ftensor f2(3, 224, 224);
  f2.Fill(1.f);
  arma::fcube cube1(224, 224, 3);
  cube1.fill(1.);
  f2.set_data(cube1);

  ASSERT_TRUE(arma::approx_equal(f2.data(), cube1, "absdiff", 1e-4));
}

TEST(test_tensor, empty) {
  ftensor f2;
  ASSERT_EQ(f2.empty(), true);

  ftensor f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
}

TEST(test_tensor, transform1) {

  ftensor f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
  f3.Transform([](const float& value) { return 1.f; });
  for (int i = 0; i < f3.size(); ++i) {
    ASSERT_EQ(f3.index(i), 1.f);
  }
}

TEST(test_tensor, transform2) {

  ftensor f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
  f3.Fill(1.f);
  f3.Transform([](const float& value) { return value * 2.f; });
  for (int i = 0; i < f3.size(); ++i) {
    ASSERT_EQ(f3.index(i), 2.f);
  }
}

TEST(test_tensor, clone) {

  ftensor f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
  f3.Rand();

  const auto& f4 = f3.Clone();
  assert(f4->data().memptr() != f3.data().memptr());
  ASSERT_EQ(f4->size(), f3.size());
  for (int i = 0; i < f3.size(); ++i) {
    ASSERT_EQ(f3.index(i), f4->index(i));
  }
}

TEST(test_tensor, raw_ptr) {

  ftensor f3(3, 3, 3);
  ASSERT_EQ(f3.raw_ptr(), f3.data().mem);
}

TEST(test_tensor, index1) {
  ftensor f3(3, 3, 3);
  ASSERT_EQ(f3.empty(), false);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(1);
  }
  f3.Fill(values);
  for (int i = 0; i < 27; ++i) {
    ASSERT_EQ(f3.index(i), 1);
  }
}

TEST(test_tensor, index2) {
  ftensor f3(3, 3, 3);
  f3.index(3) = 4;
  ASSERT_EQ(f3.index(3), 4);
}

TEST(test_tensor, flatten1) {

  ftensor f3(3, 3, 3);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.Fill(values);
  f3.Flatten(false);
  ASSERT_EQ(f3.channels(), 1);
  ASSERT_EQ(f3.rows(), 27);
  ASSERT_EQ(f3.cols(), 1);
  ASSERT_EQ(f3.index(0), 0);
  ASSERT_EQ(f3.index(1), 3);
  ASSERT_EQ(f3.index(2), 6);

  ASSERT_EQ(f3.index(3), 1);
  ASSERT_EQ(f3.index(4), 4);
  ASSERT_EQ(f3.index(5), 7);

  ASSERT_EQ(f3.index(6), 2);
  ASSERT_EQ(f3.index(7), 5);
  ASSERT_EQ(f3.index(8), 8);
}

TEST(test_tensor, flatten2) {

  ftensor f3(3, 3, 3);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.Fill(values);
  f3.Flatten(true);
  for (int i = 0; i < 27; ++i) {
    ASSERT_EQ(f3.index(i), i);
  }
}

TEST(test_tensor, fill1) {

  ftensor f3(3, 3, 3);
  std::vector<float> values;
  for (int i = 0; i < 27; ++i) {
    values.push_back(float(i));
  }
  f3.Fill(values);
  int index = 0;
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < f3.rows(); ++i) {
      for (int j = 0; j < f3.cols(); ++j) {
        ASSERT_EQ(f3.at(c, i, j), index);
        index += 1;
      }
    }
  }
}

TEST(test_tensor, create1) {
  const std::shared_ptr<ftensor>& tensor_ptr = TensorCreate(3, 32, 32);
  ASSERT_EQ(tensor_ptr->empty(), false);
  ASSERT_EQ(tensor_ptr->channels(), 3);
  ASSERT_EQ(tensor_ptr->rows(), 32);
  ASSERT_EQ(tensor_ptr->cols(), 32);
}

TEST(test_tensor, create2) {
  const std::shared_ptr<ftensor>& tensor_ptr = TensorCreate({3, 32, 32});
  ASSERT_EQ(tensor_ptr->empty(), false);
  ASSERT_EQ(tensor_ptr->channels(), 3);
  ASSERT_EQ(tensor_ptr->rows(), 32);
  ASSERT_EQ(tensor_ptr->cols(), 32);
}

TEST(test_tensor, tensor_broadcast1) {
  const std::shared_ptr<ftensor>& tensor1 = TensorCreate({3, 1, 1});
  const std::shared_ptr<ftensor>& tensor2 = TensorCreate({3, 32, 32});

  const auto& [tensor11, tensor21] = TensorBroadcast(tensor1, tensor2);
  ASSERT_EQ(tensor21->channels(), 3);
  ASSERT_EQ(tensor21->rows(), 32);
  ASSERT_EQ(tensor21->cols(), 32);

  ASSERT_EQ(tensor11->channels(), 3);
  ASSERT_EQ(tensor11->rows(), 32);
  ASSERT_EQ(tensor11->cols(), 32);

  ASSERT_TRUE(
      arma::approx_equal(tensor21->data(), tensor2->data(), "absdiff", 1e-4));
}

TEST(test_tensor, tensor_broadcast2) {
  const std::shared_ptr<ftensor>& tensor1 = TensorCreate({3, 32, 32});
  const std::shared_ptr<ftensor>& tensor2 = TensorCreate({3, 1, 1});
  tensor2->Rand();

  const auto& [tensor11, tensor21] = TensorBroadcast(tensor1, tensor2);
  ASSERT_EQ(tensor21->channels(), 3);
  ASSERT_EQ(tensor21->rows(), 32);
  ASSERT_EQ(tensor21->cols(), 32);

  ASSERT_EQ(tensor11->channels(), 3);
  ASSERT_EQ(tensor11->rows(), 32);
  ASSERT_EQ(tensor11->cols(), 32);

  for (uint32_t i = 0; i < tensor21->channels(); ++i) {
    float c = tensor2->at(i, 0, 0);
    const auto& in_channel = tensor21->slice(i);
    for (uint32_t j = 0; j < in_channel.size(); ++j) {
      ASSERT_EQ(in_channel.at(j), c);
    }
  }
}

TEST(test_tensor, fill2) {

  ftensor f3(3, 3, 3);
  f3.Fill(1.f);
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < f3.rows(); ++i) {
      for (int j = 0; j < f3.cols(); ++j) {
        ASSERT_EQ(f3.at(c, i, j), 1.f);
      }
    }
  }
}

TEST(test_tensor, add1) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(1.f);
  const auto& f2 = std::make_shared<ftensor>(3, 224, 224);
  f2->Fill(2.f);
  const auto& f3 = TensorElementAdd(f2, f1);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 3.f);
  }
}

TEST(test_tensor, add2) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(1.f);
  const auto& f2 = std::make_shared<ftensor>(3, 1, 1);
  f2->Fill(2.f);
  const auto& f3 = TensorElementAdd(f2, f1);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 3.f);
  }
}

TEST(test_tensor, add3) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(1.f);
  const auto& f2 = std::make_shared<ftensor>(3, 1, 1);
  f2->Fill(2.f);
  const auto& f3 = TensorElementAdd(f1, f2);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 3.f);
  }
}

TEST(test_tensor, add4) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(1.f);
  const auto& f2 = std::make_shared<ftensor>(3, 224, 224);
  f2->Fill(2.f);
  const auto& f3 = TensorElementAdd(f1, f2);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 3.f);
  }
}

TEST(test_tensor, mul1) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(3.f);
  const auto& f2 = std::make_shared<ftensor>(3, 224, 224);
  f2->Fill(2.f);
  const auto& f3 = TensorElementMultiply(f2, f1);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 6.f);
  }
}

TEST(test_tensor, mul2) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(3.f);
  const auto& f2 = std::make_shared<ftensor>(3, 1, 1);
  f2->Fill(2.f);
  const auto& f3 = TensorElementMultiply(f2, f1);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 6.f);
  }
}

TEST(test_tensor, mul3) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(3.f);
  const auto& f2 = std::make_shared<ftensor>(3, 224, 224);
  f2->Fill(2.f);
  const auto& f3 = TensorElementMultiply(f1, f2);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 6.f);
  }
}

TEST(test_tensor, mul4) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(3.f);
  const auto& f2 = std::make_shared<ftensor>(3, 1, 1);
  f2->Fill(2.f);
  const auto& f3 = TensorElementMultiply(f1, f2);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 6.f);
  }
}

TEST(test_tensor, mul5) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(3.f);
  const auto& f2 = std::make_shared<ftensor>(3, 1, 1);
  f2->Fill(2.f);

  const auto& f3 = std::make_shared<ftensor>(3, 224, 224);
  TensorElementMultiply(f1, f2, f3);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 6.f);
  }
}

TEST(test_tensor, mul6) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(3.f);
  const auto& f2 = std::make_shared<ftensor>(3, 1, 1);
  f2->Fill(2.f);

  const auto& f3 = std::make_shared<ftensor>(3, 224, 224);
  TensorElementMultiply(f2, f1, f3);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 6.f);
  }
}

TEST(test_tensor, add5) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(3.f);
  const auto& f2 = std::make_shared<ftensor>(3, 1, 1);
  f2->Fill(2.f);

  const auto& f3 = std::make_shared<ftensor>(3, 224, 224);
  TensorElementAdd(f1, f2, f3);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 5.f);
  }
}

TEST(test_tensor, add6) {
  const auto& f1 = std::make_shared<ftensor>(3, 224, 224);
  f1->Fill(3.f);
  const auto& f2 = std::make_shared<ftensor>(3, 1, 1);
  f2->Fill(2.f);

  const auto& f3 = std::make_shared<ftensor>(3, 224, 224);
  TensorElementAdd(f2, f1, f3);
  for (int i = 0; i < f3->size(); ++i) {
    ASSERT_EQ(f3->index(i), 5.f);
  }
}

TEST(test_tensor, shapes) {
  ftensor f3(2, 3, 4);
  const std::vector<uint32_t>& shapes = f3.shape();
  ASSERT_EQ(shapes.at(0), 2);
  ASSERT_EQ(shapes.at(1), 3);
  ASSERT_EQ(shapes.at(2), 4);
}

TEST(test_tensor, raw_shapes1) {
  ftensor f3(2, 3, 4);
  f3.Reshape({24});
  const auto& shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 1);
  ASSERT_EQ(shapes.at(0), 24);
}

TEST(test_tensor, raw_shapes2) {
  ftensor f3(2, 3, 4);
  f3.Reshape({4, 6});
  const auto& shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 2);
  ASSERT_EQ(shapes.at(0), 4);
  ASSERT_EQ(shapes.at(1), 6);
}

TEST(test_tensor, raw_shapes3) {
  ftensor f3(2, 3, 4);
  f3.Reshape({4, 3, 2});
  const auto& shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 3);
  ASSERT_EQ(shapes.at(0), 4);
  ASSERT_EQ(shapes.at(1), 3);
  ASSERT_EQ(shapes.at(2), 2);
}

TEST(test_tensor, raw_view1) {
  ftensor f3(2, 3, 4);
  f3.Reshape({24}, true);
  const auto& shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 1);
  ASSERT_EQ(shapes.at(0), 24);
}

TEST(test_tensor, raw_view2) {
  ftensor f3(2, 3, 4);
  f3.Reshape({4, 6}, true);
  const auto& shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 2);
  ASSERT_EQ(shapes.at(0), 4);
  ASSERT_EQ(shapes.at(1), 6);
}

TEST(test_tensor, raw_view3) {
  ftensor f3(2, 3, 4);
  f3.Reshape({4, 3, 2}, true);
  const auto& shapes = f3.raw_shapes();
  ASSERT_EQ(shapes.size(), 3);
  ASSERT_EQ(shapes.at(0), 4);
  ASSERT_EQ(shapes.at(1), 3);
  ASSERT_EQ(shapes.at(2), 2);
}

TEST(test_tensor, padding1) {
  ftensor tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);
  tensor.Padding({1, 2, 3, 4}, 0);
  ASSERT_EQ(tensor.rows(), 7);
  ASSERT_EQ(tensor.cols(), 12);

  int index = 0;
  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        if ((r >= 2 && r <= 4) && (c_ >= 3 && c_ <= 7)) {
          ASSERT_EQ(tensor.at(c, r, c_), 1.f) << c << " "
                                              << " " << r << " " << c_;
        }
        index += 1;
      }
    }
  }
}

TEST(test_tensor, padding2) {
  ftensor tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);
  tensor.Padding({2, 2, 2, 2}, 3.14f);
  ASSERT_EQ(tensor.rows(), 8);
  ASSERT_EQ(tensor.cols(), 9);

  int index = 0;
  for (int c = 0; c < tensor.channels(); ++c) {
    for (int r = 0; r < tensor.rows(); ++r) {
      for (int c_ = 0; c_ < tensor.cols(); ++c_) {
        if (c_ <= 1 || r <= 1) {
          ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
        } else if (c >= tensor.cols() - 1 || r >= tensor.rows() - 1) {
          ASSERT_EQ(tensor.at(c, r, c_), 3.14f);
        }
        if ((r >= 2 && r <= 5) && (c_ >= 2 && c_ <= 6)) {
          ASSERT_EQ(tensor.at(c, r, c_), 1.f);
        }
        index += 1;
      }
    }
  }
}

TEST(test_tensor, review1) {
  ftensor tensor(3, 4, 5);
  std::vector<float> values;
  for (int i = 0; i < 60; ++i) {
    values.push_back(float(i));
  }

  tensor.Fill(values);

  tensor.Reshape({4, 3, 5}, true);
  auto data = tensor.slice(0);
  int index = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      ASSERT_EQ(data.at(i, j), index);
      index += 1;
    }
  }
  data = tensor.slice(1);
  index = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      ASSERT_EQ(data.at(i, j), index + 15);
      index += 1;
    }
  }
  index = 0;
  data = tensor.slice(2);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      ASSERT_EQ(data.at(i, j), index + 30);
      index += 1;
    }
  }

  index = 0;
  data = tensor.slice(3);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      ASSERT_EQ(data.at(i, j), index + 45);
      index += 1;
    }
  }
}

TEST(test_tensor, review2) {
  arma::fmat f1 =
      "1,2,3,4;"
      "5,6,7,8";

  arma::fmat f2 =
      "1,2,3,4;"
      "5,6,7,8";
  sftensor data = TensorCreate(2, 2, 4);
  data->slice(0) = f1;
  data->slice(1) = f2;
  data->Reshape({16}, true);
  for (uint32_t i = 0; i < 8; ++i) {
    ASSERT_EQ(data->index(i), i + 1);
  }

  for (uint32_t i = 8; i < 15; ++i) {
    ASSERT_EQ(data->index(i - 8), i - 8 + 1);
  }
}

TEST(test_tensor, review3) {
  arma::fmat f1 =
      "1,2,3,4;"
      "5,6,7,8";

  sftensor data = TensorCreate(1, 2, 4);
  data->slice(0) = f1;
  data->Reshape({4, 2}, true);

  arma::fmat data2 = data->slice(0);
  ASSERT_EQ(data2.n_rows, 4);
  ASSERT_EQ(data2.n_cols, 2);
  uint32_t index = 1;
  for (uint32_t row = 0; row < data2.n_rows; ++row) {
    for (uint32_t col = 0; col < data2.n_cols; ++col) {
      ASSERT_EQ(data2.at(row, col), index);
      index += 1;
    }
  }
}

TEST(test_tensor, review4) {
  arma::fmat f1 =
      "1,2,3,4;"
      "5,6,7,8";

  arma::fmat f2 =
      "9,10,11,12;"
      "13,14,15,16";

  sftensor data = TensorCreate(2, 2, 4);
  data->slice(0) = f1;
  data->slice(1) = f2;
  data->Reshape({4, 2, 2}, true);
  for (uint32_t c = 0; c < data->channels(); ++c) {
    const auto& in_channel = data->slice(c);
    ASSERT_EQ(in_channel.n_rows, 2);
    ASSERT_EQ(in_channel.n_cols, 2);
    float n1 = in_channel.at(0, 0);
    float n2 = in_channel.at(0, 1);
    float n3 = in_channel.at(1, 0);
    float n4 = in_channel.at(1, 1);
    ASSERT_EQ(n1, c * 4 + 1);
    ASSERT_EQ(n2, c * 4 + 2);
    ASSERT_EQ(n3, c * 4 + 3);
    ASSERT_EQ(n4, c * 4 + 4);
  }
}

TEST(test_tensor, reshape1) {
  arma::fmat f1 =
      "1,3;"
      "2,4";

  arma::fmat f2 =
      "1,3;"
      "2,4";

  sftensor data = TensorCreate(2, 2, 2);
  data->slice(0) = f1;
  data->slice(1) = f2;
  data->Reshape({8});
  for (uint32_t i = 0; i < 4; ++i) {
    ASSERT_EQ(data->index(i), i + 1);
  }

  for (uint32_t i = 4; i < 8; ++i) {
    ASSERT_EQ(data->index(i - 4), i - 4 + 1);
  }
}

TEST(test_tensor, reshape2) {
  arma::fmat f1 =
      "0,2;"
      "1,3";

  arma::fmat f2 =
      "0,2;"
      "1,3";

  sftensor data = TensorCreate(2, 2, 2);
  data->slice(0) = f1;
  data->slice(1) = f2;
  data->Reshape({2, 4});
  for (uint32_t i = 0; i < 4; ++i) {
    ASSERT_EQ(data->index(i), i);
  }

  for (uint32_t i = 4; i < 8; ++i) {
    ASSERT_EQ(data->index(i), i - 4);
  }
}

TEST(test_tensor, ones) {
  ftensor tensor(3, 4, 5);
  tensor.Ones();
  for (int i = 0; i < tensor.size(); ++i) {
    ASSERT_EQ(tensor.index(i), 1.f);
  }
}

TEST(test_tensor, rand) {
  ftensor tensor(3, 4, 5);
  tensor.Fill(99.f);
  tensor.Rand();  // 0 ~ 1
  for (int i = 0; i < tensor.size(); ++i) {
    ASSERT_NE(tensor.index(i), 99.f);
  }
}

TEST(test_tensor, get_data) {
  ftensor tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  arma::fmat in2(4, 5);
  const arma::fmat& in1 = tensor.slice(0);
  tensor.slice(0) = in2;
  const arma::fmat& in3 = tensor.slice(0);
  ASSERT_EQ(in1.memptr(), in3.memptr());
}

TEST(test_tensor, at1) {
  ftensor tensor(3, 4, 5);
  tensor.at(0, 1, 2) = 2;
  ASSERT_EQ(tensor.at(0, 1, 2), 2);
}

TEST(test_tensor, at2) {
  ftensor tensor(3, 4, 5);
  arma::fmat f(4, 5);
  f.fill(1.f);
  tensor.slice(0) = f;
  ASSERT_TRUE(arma::approx_equal(f, tensor.slice(0), "absdiff", 1e-4));
}

TEST(test_tensor, at3) {
  ftensor tensor(3, 4, 5);
  tensor.Fill(1.2f);
  for (uint32_t c = 0; c < tensor.channels(); ++c) {
    for (uint32_t r = 0; r < tensor.rows(); ++r) {
      for (uint32_t c_ = 0; c_ < tensor.cols(); ++c_) {
        ASSERT_EQ(tensor.at(c, r, c_), 1.2f);
      }
    }
  }
}

TEST(test_tensor, is_same1) {
  std::shared_ptr<ftensor> in1 =
      std::make_shared<ftensor>(3, 32, 32);
  in1->Fill(2.f);

  std::shared_ptr<ftensor> in2 =
      std::make_shared<ftensor>(3, 32, 32);
  in2->Fill(2.f);
  ASSERT_EQ(TensorIsSame(in1, in2), true);
}

TEST(test_tensor, is_same2) {
  std::shared_ptr<ftensor> in1 =
      std::make_shared<ftensor>(3, 32, 32);
  in1->Fill(1.f);

  std::shared_ptr<ftensor> in2 =
      std::make_shared<ftensor>(3, 32, 32);
  in2->Fill(2.f);
  ASSERT_EQ(TensorIsSame(in1, in2), false);
}

TEST(test_tensor, is_same3) {
  std::shared_ptr<ftensor> in1 =
      std::make_shared<ftensor>(3, 32, 32);
  in1->Fill(1.f);

  std::shared_ptr<ftensor> in2 =
      std::make_shared<ftensor>(3, 31, 32);
  in2->Fill(1.f);
  ASSERT_EQ(TensorIsSame(in1, in2), false);
}

TEST(test_tensor, tensor_padding1) {
  sftensor tensor = TensorCreate(3, 4, 5);
  ASSERT_EQ(tensor->channels(), 3);
  ASSERT_EQ(tensor->rows(), 4);
  ASSERT_EQ(tensor->cols(), 5);

  tensor->Fill(1.f);
  tensor = TensorPadding(tensor, {2, 2, 2, 2}, 3.14f);
  ASSERT_EQ(tensor->rows(), 8);
  ASSERT_EQ(tensor->cols(), 9);

  int index = 0;
  for (int c = 0; c < tensor->channels(); ++c) {
    for (int r = 0; r < tensor->rows(); ++r) {
      for (int c_ = 0; c_ < tensor->cols(); ++c_) {
        if (c_ <= 1 || r <= 1) {
          ASSERT_EQ(tensor->at(c, r, c_), 3.14f);
        } else if (c >= tensor->cols() - 1 || r >= tensor->rows() - 1) {
          ASSERT_EQ(tensor->at(c, r, c_), 3.14f);
        }
        if ((r >= 2 && r <= 5) && (c_ >= 2 && c_ <= 6)) {
          ASSERT_EQ(tensor->at(c, r, c_), 1.f);
        }
        index += 1;
      }
    }
  }
}