#include "runtime/runtime_ir.hpp"
#include <gtest/gtest.h>

using namespace TinyInfer;

TEST(test_runtime, runtime_graph_input_init1) {
  std::vector<srunop> ops;
  uint32_t op_size = 3;
  for (uint32_t i = 0; i < op_size; ++i) {
    const auto &op = std::make_shared<RuntimeOperator>();
    const auto &oprand1 = std::make_shared<RuntimeOperand>();
    oprand1->shape = {3, 32, 32};
    oprand1->type = RuntimeDataType::TypeFloat32;

    const auto &oprand2 = std::make_shared<RuntimeOperand>();
    oprand2->shape = {3, 64, 64};
    oprand2->type = RuntimeDataType::TypeFloat32;

    op->in_oprands.insert({std::string("batch1"), oprand1});
    op->in_oprands.insert({std::string("batch2"), oprand2});

    ops.push_back(op);
  }

  ASSERT_EQ(ops.size(), op_size);
  RuntimeOperatorUtils::InitOpsInput(ops);
  for (uint32_t i = 0; i < ops.size(); ++i) {
    const auto &op = ops.at(i);
    ASSERT_EQ(op->in_oprands["batch1"]->data.empty(), false);
    const uint32_t batch1 = op->in_oprands["batch1"]->data.size();
    const uint32_t batch2 = op->in_oprands["batch2"]->data.size();
    ASSERT_EQ(batch1, 3);
    ASSERT_EQ(batch1, batch2);
  }

  RuntimeOperatorUtils::InitOpsInput(ops);
  for (uint32_t i = 0; i < ops.size(); ++i) {
    const auto &op = ops.at(i);
    ASSERT_EQ(op->in_oprands["batch1"]->data.empty(), false);
    const uint32_t batch1 = op->in_oprands["batch1"]->data.size();
    const uint32_t batch2 = op->in_oprands["batch2"]->data.size();
    ASSERT_EQ(batch1, 3);
    ASSERT_EQ(batch1, batch2);
  }
}

TEST(test_runtime, runtime_graph_input_init2) {
  std::vector<srunop> ops;
  uint32_t op_size = 3;
  for (uint32_t i = 0; i < op_size; ++i) {
    const auto &op = std::make_shared<RuntimeOperator>();
    const auto &oprand1 = std::make_shared<RuntimeOperand>();
    oprand1->shape = {4, 3, 32, 32};
    oprand1->type = RuntimeDataType::TypeFloat32;

    const auto &oprand2 = std::make_shared<RuntimeOperand>();
    oprand2->shape = {4, 3, 64, 64};
    oprand2->type = RuntimeDataType::TypeFloat32;

    op->in_oprands.insert({std::string("batch1"), oprand1});
    op->in_oprands.insert({std::string("batch2"), oprand2});

    ops.push_back(op);
  }

  ASSERT_EQ(ops.size(), op_size);
  RuntimeOperatorUtils::InitOpsInput(ops);
  for (uint32_t i = 0; i < ops.size(); ++i) {
    const auto &op = ops.at(i);
    ASSERT_EQ(op->in_oprands["batch1"]->data.empty(), false);
    const uint32_t batch1 = op->in_oprands["batch1"]->data.size();
    const uint32_t batch2 = op->in_oprands["batch2"]->data.size();
    ASSERT_EQ(batch1, 4);
    ASSERT_EQ(batch1, batch2);
  }

  RuntimeOperatorUtils::InitOpsInput(ops);
  for (uint32_t i = 0; i < ops.size(); ++i) {
    const auto &op = ops.at(i);
    ASSERT_EQ(op->in_oprands["batch1"]->data.empty(), false);
    const uint32_t batch1 = op->in_oprands["batch1"]->data.size();
    const uint32_t batch2 = op->in_oprands["batch2"]->data.size();
    ASSERT_EQ(batch1, 4);
    ASSERT_EQ(batch1, batch2);
  }
}

TEST(test_runtime, runtime_graph_input_init3) {
  std::vector<srunop> ops;
  uint32_t op_size = 3;
  for (uint32_t i = 0; i < op_size; ++i) {
    const auto &op = std::make_shared<RuntimeOperator>();
    const auto &oprand1 = std::make_shared<RuntimeOperand>();
    oprand1->shape = {5, 32};
    oprand1->type = RuntimeDataType::TypeFloat32;

    const auto &oprand2 = std::make_shared<RuntimeOperand>();
    oprand2->shape = {5, 64};
    oprand2->type = RuntimeDataType::TypeFloat32;

    op->in_oprands.insert({std::string("batch1"), oprand1});
    op->in_oprands.insert({std::string("batch2"), oprand2});

    ops.push_back(op);
  }

  ASSERT_EQ(ops.size(), op_size);
  RuntimeOperatorUtils::InitOpsInput(ops);
  for (uint32_t i = 0; i < ops.size(); ++i) {
    const auto &op = ops.at(i);
    ASSERT_EQ(op->in_oprands["batch1"]->data.empty(), false);
    const uint32_t batch1 = op->in_oprands["batch1"]->data.size();
    const uint32_t batch2 = op->in_oprands["batch2"]->data.size();
    ASSERT_EQ(batch1, 5);
    ASSERT_EQ(batch1, batch2);
  }

  RuntimeOperatorUtils::InitOpsInput(ops);
  for (uint32_t i = 0; i < ops.size(); ++i) {
    const auto &op = ops.at(i);
    ASSERT_EQ(op->in_oprands["batch1"]->data.empty(), false);
    const uint32_t batch1 = op->in_oprands["batch1"]->data.size();
    const uint32_t batch2 = op->in_oprands["batch2"]->data.size();
    ASSERT_EQ(batch1, 5);
    ASSERT_EQ(batch1, batch2);
  }
}

TEST(test_runtime, runtime_graph_output_init1) {
  std::vector<pnnx::Operator *> pnnx_ops;
  std::vector<srunop> ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator *pnnx_op = new pnnx::Operator;
    pnnx::Operand *pnnx_oprand = new pnnx::Operand;
    pnnx_oprand->shape = std::vector<int>{8, 3, 32, 32};
    pnnx_op->outputs.push_back(pnnx_oprand);
    pnnx_ops.push_back(pnnx_op);
    ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils::InitOpsOutput(pnnx_ops, ops);

  for (const auto &op : ops) {
    const auto &out_oprand = op->out_oprand;
    ASSERT_EQ(out_oprand->shape.size(), 4);
    ASSERT_EQ(out_oprand->data.size(), 8);
    for (const auto &out_tensor : out_oprand->data) {
      ASSERT_EQ(out_tensor->rows(), 32);
      ASSERT_EQ(out_tensor->cols(), 32);
      ASSERT_EQ(out_tensor->channels(), 3);
      out_tensor->data().resize(32, 16, 6); // Reshape输出Tensor的维度
    }
  }

  RuntimeOperatorUtils::InitOpsOutput(
      pnnx_ops, ops); // 注意：这其中会恢复输出Tensor的维度

  for (const auto &op : ops) {
    const auto &out_oprand = op->out_oprand;
    ASSERT_EQ(out_oprand->shape.size(), 4);
    ASSERT_EQ(out_oprand->data.size(), 8);
    for (const auto &out_tensor : out_oprand->data) {
      ASSERT_EQ(out_tensor->rows(), 32);
      ASSERT_EQ(out_tensor->cols(), 32);
      ASSERT_EQ(out_tensor->channels(), 3);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init2) {
  std::vector<pnnx::Operator *> pnnx_ops;
  std::vector<srunop> ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator *pnnx_op = new pnnx::Operator;
    pnnx::Operand *pnnx_oprand = new pnnx::Operand;
    pnnx_oprand->shape = std::vector<int>{8, 64};
    pnnx_op->outputs.push_back(pnnx_oprand);
    pnnx_ops.push_back(pnnx_op);
    ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils::InitOpsOutput(pnnx_ops, ops);

  for (const auto &op : ops) {
    const auto &out_oprand = op->out_oprand;
    ASSERT_EQ(out_oprand->shape.size(), 2);
    ASSERT_EQ(out_oprand->data.size(), 8);

    for (const auto &out_tensor : out_oprand->data) {
      ASSERT_EQ(out_tensor->rows(), 64);
      ASSERT_EQ(out_tensor->cols(), 1);
      ASSERT_EQ(out_tensor->channels(), 1);
      out_tensor->data().resize(32, 1, 2);
    }
  }

  RuntimeOperatorUtils::InitOpsOutput(pnnx_ops, ops);

  for (const auto &op : ops) {
    const auto &out_oprand = op->out_oprand;
    ASSERT_EQ(out_oprand->shape.size(), 2);
    ASSERT_EQ(out_oprand->data.size(), 8);

    for (const auto &out_tensor : out_oprand->data) {
      ASSERT_EQ(out_tensor->rows(), 64);
      ASSERT_EQ(out_tensor->cols(), 1);
      ASSERT_EQ(out_tensor->channels(), 1);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init3) {
  std::vector<pnnx::Operator *> pnnx_ops;
  std::vector<srunop> ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator *pnnx_op = new pnnx::Operator;
    pnnx::Operand *pnnx_oprand = new pnnx::Operand;
    pnnx_oprand->shape = std::vector<int>{8, 32, 32};
    pnnx_op->outputs.push_back(pnnx_oprand);
    pnnx_ops.push_back(pnnx_op);
    ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils::InitOpsOutput(pnnx_ops, ops);

  for (const auto &op : ops) {
    const auto &out_oprand = op->out_oprand;
    ASSERT_EQ(out_oprand->shape.size(), 3);
    ASSERT_EQ(out_oprand->data.size(), 8);
    for (const auto &out_tensor : out_oprand->data) {
      ASSERT_EQ(out_tensor->rows(), 32);
      ASSERT_EQ(out_tensor->cols(), 32);
      ASSERT_EQ(out_tensor->channels(), 1);
      out_tensor->data().resize(32, 1, 32);
    }
  }

  RuntimeOperatorUtils::InitOpsOutput(pnnx_ops, ops);

  for (const auto &op : ops) {
    const auto &out_oprand = op->out_oprand;
    ASSERT_EQ(out_oprand->shape.size(), 3);
    ASSERT_EQ(out_oprand->data.size(), 8);
    for (const auto &out_tensor : out_oprand->data) {
      ASSERT_EQ(out_tensor->rows(), 32);
      ASSERT_EQ(out_tensor->cols(), 32);
      ASSERT_EQ(out_tensor->channels(), 1);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init4) {
  std::vector<pnnx::Operator *> pnnx_ops;
  std::vector<srunop> ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator *pnnx_op = new pnnx::Operator;
    pnnx::Operand *pnnx_oprand = new pnnx::Operand;
    pnnx_oprand->shape = std::vector<int>{8, 32, 32};
    pnnx_op->outputs.push_back(pnnx_oprand);
    pnnx_ops.push_back(pnnx_op);
    ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils::InitOpsOutput(pnnx_ops, ops);

  for (const auto &op : ops) {
    const auto &out_oprand = op->out_oprand;
    ASSERT_EQ(out_oprand->shape.size(), 3);
    ASSERT_EQ(out_oprand->data.size(), 8);
    for (auto &out_tensor : out_oprand->data) {
      ASSERT_EQ(out_tensor->rows(), 32);
      ASSERT_EQ(out_tensor->cols(), 32);
      ASSERT_EQ(out_tensor->channels(), 1);
      out_tensor->data().resize(32 * 32, 1, 1);
    }
  }

  RuntimeOperatorUtils::InitOpsOutput(pnnx_ops, ops);

  for (const auto &op : ops) {
    const auto &out_oprand = op->out_oprand;
    ASSERT_EQ(out_oprand->shape.size(), 3);
    ASSERT_EQ(out_oprand->data.size(), 8);
    for (const auto &out_tensor : out_oprand->data) {
      ASSERT_EQ(out_tensor->rows(), 32);
      ASSERT_EQ(out_tensor->cols(), 32);
      ASSERT_EQ(out_tensor->channels(), 1);
    }
  }
}

TEST(test_runtime, set_param_path) {
  RuntimeGraph graph("xx.param", "yy.bin");
  ASSERT_EQ(graph.param_path(), "xx.param");
  graph.set_param_path("yy.param");
  ASSERT_EQ(graph.param_path(), "yy.param");
}

TEST(test_runtime, set_bin_path) {
  RuntimeGraph graph("xx.param", "yy.bin");
  ASSERT_EQ(graph.bin_path(), "yy.bin");
  graph.set_bin_path("yy.bin");
  ASSERT_EQ(graph.bin_path(), "yy.bin");
}