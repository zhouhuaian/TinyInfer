#include "runtime/runtime_op.hpp"
#include <array>

namespace TinyInfer {

RuntimeOp::~RuntimeOp() {
  for (auto &param : this->params) {
    if (param.second != nullptr) {
      delete param.second;
      param.second = nullptr;
    }
  }
}

void RuntimeOpUtils::InitOpsInput(const std::vector<srunop> &ops) {
  CHECK(!ops.empty()) << "Operators for init input shapes is empty!";

  for (const auto &op : ops) {
    if (op->in_oprands.empty()) {
      continue;
    } else {
      const auto &in_oprands_map = op->in_oprands;
      // 遍历输入操作数
      for (const auto &[name, in_oprand] : in_oprands_map) {
        // 检查输入操作数的值类型
        const auto &type = in_oprand->type;
        CHECK(type == RuntimeDataType::TypeFloat32)
            << "TinyInfer only supports float32 yet!";

        // 检查输入操作数的维度
        const auto &in_shape = in_oprand->shape;
        CHECK(!in_shape.empty());
        const int32_t batch = in_shape.at(0);
        CHECK(batch >= 0) << "Dynamic batch size is not supported!";
        CHECK(in_shape.size() == 2 || in_shape.size() == 3 ||
              in_shape.size() == 4)
            << "Unsupported input oprand shape size: " << in_shape.size();

        // 取出表示输入操作数的Tensor数组
        auto &input_data = in_oprand->data;
        // 数组非空，说明计算图已经构建完毕，检查其size是否等于batch
        if (!input_data.empty()) {
          CHECK(input_data.size() == batch)
              << "Input tensor count not equal to batch!";
        }
        // 数组为空，说明是第一次构建计算图，则预留数组的容量等于batch
        else {
          input_data.resize(batch);
        }
      }
    }
  }
}

void RuntimeOpUtils::InitOpsOutput(
    const std::vector<pnnx::Operator *> &pnnx_ops,
    const std::vector<srunop> &ops) {
  CHECK(!pnnx_ops.empty() && !ops.empty());
  CHECK(pnnx_ops.size() == ops.size());

  // 由pnnx计算图节点的输出操作数初始化TinyInfer计算图节点的输出空间
  for (uint32_t i = 0; i < pnnx_ops.size(); ++i) {
    // 获取pnnx计算图节点的输出操作数
    const auto &pout_oprands = pnnx_ops.at(i)->outputs;

    // 一个节点最多只有一个输出操作数
    if (pout_oprands.empty()) {
      continue;
    }
    CHECK(pout_oprands.size() == 1)
        << "One operator has <= one output oprand in TinyInfer";

    const auto pout_oprand = pout_oprands.front();
    CHECK(pout_oprand != nullptr) << "Output oprand is null";

    // 检查输出操作数的维度
    const std::vector<int32_t> &out_shape = pout_oprand->shape;
    const int32_t batch = out_shape.at(0);
    CHECK(batch >= 0) << "Dynamic batch size is not supported!";
    CHECK(out_shape.size() == 2 || out_shape.size() == 3 ||
          out_shape.size() == 4)
        << "Unsupported output oprand shape size: " << out_shape.size();

    // 取出需要初始化的输出操作数
    const auto &op = ops.at(i);
    const auto &out_oprand = op->out_oprand;

    // 输出操作数为空，说明是第一次构建计算图
    if (!out_oprand) {
      // 初始化输出操作数的名称、维度、值类型
      srunoprand tmp_out_oprand = std::make_shared<RuntimeOprand>();
      tmp_out_oprand->name = pout_oprand->name + "_output";
      tmp_out_oprand->shape = out_shape;
      tmp_out_oprand->type = RuntimeDataType::TypeFloat32;
      tmp_out_oprand->data.reserve(batch);

      // 初始化输出空间——开辟保存每一个输出Tensor的内存空间
      // 注意：初始化输入空间时，不用为输入Tensor开辟空间，因为输入Tensor一定是从前驱节点接收来的！
      for (int b = 0; b < batch; ++b) {
        if (out_shape.size() == 2) {
          tmp_out_oprand->data.push_back(
              std::make_shared<ftensor>(1, out_shape.at(1), 1));
        } else if (out_shape.size() == 3) {
          tmp_out_oprand->data.push_back(
              std::make_shared<ftensor>(1, out_shape.at(1), out_shape.at(2)));
        } else { // current shape size is 4
          tmp_out_oprand->data.push_back(std::make_shared<ftensor>(
              out_shape.at(1), out_shape.at(2), out_shape.at(3)));
        }
      }
      op->out_oprand = std::move(tmp_out_oprand);
    }

    // 输出操作数非空，说明计算图已经构建完毕，则检查
    else {
      // 检查输出操作数的维度、值类型
      CHECK(out_oprand->shape == out_shape);
      CHECK(out_oprand->type == RuntimeDataType::TypeFloat32);

      // 检查每一个输出Tensor是否变形，若变形则需要恢复
      for (uint32_t b = 0; b < batch; ++b) {
        const auto &tensor_shape = out_oprand->data.at(b)->shape();

        if (out_shape.size() == 2) {
          if (tensor_shape.at(0) != 1 ||
              tensor_shape.at(1) != out_shape.at(1) ||
              tensor_shape.at(2) != 1) {
            DLOG(WARNING) << "The shape of output tensor do not adapting with "
                             "output oprand shape";
            const auto &target_shape =
                std::vector<uint32_t>{1, (uint32_t)out_shape.at(1), 1};
            out_oprand->data.at(b)->Reshape(target_shape);
          }
        } else if (out_shape.size() == 3) {
          if (tensor_shape.at(0) != 1 ||
              tensor_shape.at(1) != out_shape.at(1) ||
              tensor_shape.at(2) != out_shape.at(2)) {
            DLOG(WARNING) << "The shape of output tensor do not adapting with "
                             "output oprand shape";
            const auto &target_shape = std::vector<uint32_t>{
                1, (uint32_t)out_shape.at(1), (uint32_t)out_shape.at(2)};
            out_oprand->data.at(b)->Reshape(target_shape);
          }
        } else { // current shape size is 4
          if (tensor_shape.at(0) != out_shape.at(1) ||
              tensor_shape.at(1) != out_shape.at(2) ||
              tensor_shape.at(2) != out_shape.at(3)) {
            DLOG(WARNING) << "The shape of output tensor do not adapting with "
                             "output oprand shape";
            const auto &target_shape = std::vector<uint32_t>{
                (uint32_t)out_shape.at(1), (uint32_t)out_shape.at(2),
                (uint32_t)out_shape.at(3)};
            out_oprand->data.at(b)->Reshape(target_shape);
          }
        }
      }
    }
  }
}

} // namespace TinyInfer
