#include "softmax.hpp"
#include "data/tensor.hpp"
#include "kernel/abstract/kernel_factory.hpp"
#include <cstdint>
#include <functional>
#include <glog/logging.h>
#include <numeric>

namespace TinyInfer {

#define POS_INDEX(outer_idx, inner_idx, axis_idx)                              \
  outer_idx *axis_sz *inner_sz + axis_idx *inner_sz + inner_idx;

Softmax::Softmax(int dim) : NoAttrKernel("Softmax"), dim_(dim) {}

InferStatus Softmax::Forward(const std::vector<sftensor> &inputs,
                             std::vector<sftensor> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array is empty";
    return InferStatus::InferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  const uint32_t batch = inputs.size();

#pragma omp parallel for num_threads(batch)
  for (uint32_t b = 0; b < batch; ++b) {
    const sftensor &input = inputs.at(b);
    sftensor &output = outputs.at(b);

    CHECK(input != nullptr && !input->empty())
        << "The " << b << " input tensor is empty";

    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The " << b << " output tensor is empty";
      output = std::make_shared<ftensor>(input->shape());
    }

    CHECK(input->shape() == output->shape())
        << "The " << b << " input and output tensor shape do not match";

    int dim = this->dim_;
    std::vector<uint32_t> raw_shape = input->raw_shape();

    // 将dim转换为正数
    if (dim < 0) {
      dim += int(raw_shape.size());
    }

    // ! dim是执行softmax操作的维度在raw_shape中的位置
    CHECK(dim >= 0 && dim < raw_shape.size())
        << "Error softmax dimension: " << dim << ", which need between 0 and 2";

    // 填充欠缺的维度：如{128, 256}->{128, 256, 1}，其中1是channels
    const uint32_t padding_dim_num = 3 - raw_shape.size();
    for (uint32_t i = 0; i < padding_dim_num; ++i) {
      raw_shape.push_back(1);
    }

    // 以dim轴为界分别计算outer_sz、axis_sz、inner_sz
    const uint32_t outer_sz =
        std::accumulate(raw_shape.begin(), raw_shape.begin() + dim, 1,
                        std::multiplies<uint32_t>());
    const uint32_t inner_sz =
        std::accumulate(raw_shape.begin() + dim + 1, raw_shape.end(), 1,
                        std::multiplies<uint32_t>());
    const uint32_t axis_sz = raw_shape.at(dim);
    CHECK_EQ(outer_sz * axis_sz * inner_sz, input->size());

    // 以行优先取出输入Tensor中所有元素
    const auto in_vals = input->values(true);
    std::vector<float> out_vals(in_vals.size());

#pragma omp parallel for collapse(2) // 线程化下面两层循环
    for (uint32_t outer_idx = 0; outer_idx < outer_sz; ++outer_idx) {
      for (uint32_t inner_idx = 0; inner_idx < inner_sz; ++inner_idx) {
        // 确定dim轴上最大值
        float max_val = std::numeric_limits<float>::lowest();
        for (uint32_t axis_idx = 0; axis_idx < axis_sz; ++axis_idx) {
          // 计算元素在Tensor中的位置（行优先）
          uint32_t idx = POS_INDEX(outer_idx, inner_idx, axis_idx);
          float in_val = in_vals.at(idx);
          max_val = (in_val > max_val) ? in_val : max_val;
        }

        // ! 计算dim轴上元素x_i的exp(x_i -
        // max_val)值——添加偏移量为了避免计算指数值时溢出
        float sum_val = 0.f; // 保存dim轴上的指数值之和
        for (uint32_t axis_idx = 0; axis_idx < axis_sz; ++axis_idx) {
          uint32_t idx = POS_INDEX(outer_idx, inner_idx, axis_idx);
          float in_val = in_vals.at(idx);
          float exp_val = std::exp(in_val - max_val);

          out_vals.at(idx) = exp_val;
          sum_val += exp_val;
        }

        // 计算dim轴上元素x_i的softmax值——exp(x_i - max_val) / sum(exp(x_i -
        // max_val))
        for (uint32_t axis_idx = 0; axis_idx < axis_sz; ++axis_idx) {
          uint32_t idx = POS_INDEX(outer_idx, inner_idx, axis_idx);
          out_vals.at(idx) /= sum_val;
        }
      }
    }

    output->Fill(out_vals, true);
  }

  return InferStatus::InferSuccess;
}

ParseParamAttrStatus Softmax::Creator(const srunop &op, skernel &softmax) {
  if (op == nullptr) {
    LOG(ERROR) << "Operator is empty";
    return ParseParamAttrStatus::OpEmpty;
  }

  const auto &params = op->params;
  if (params.find("dim") == params.end() || params.at("dim") == nullptr) {
    LOG(ERROR) << "Dimension parameter is missing";
    return ParseParamAttrStatus::ParamMissingDim;
  }

  // 向下转型
  auto dim = dynamic_cast<RuntimeParamInt *>(params.at("dim"));
  if (dim == nullptr) {
    LOG(ERROR) << "Dimension parameter is missing";
    return ParseParamAttrStatus::ParamMissingDim;
  }

  softmax = std::make_shared<Softmax>(dim->value);

  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

KernelRegisterWrapper SoftmaxCreatorNN("nn.Softmax", Softmax::Creator);
KernelRegisterWrapper SoftmaxCreatorF("F.softmax", Softmax::Creator);

} // namespace TinyInfer