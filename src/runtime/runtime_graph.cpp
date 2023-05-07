#include "runtime/runtime_graph.hpp"
#include "kernel/abstract/kernel_factory.hpp"
#include "tick.hpp"
#include <algorithm>
#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TinyInfer {

RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)),
      graph_state_(GraphState::NeedInit) {}

void RuntimeGraph::set_param_path(const std::string &param_path) {
  this->param_path_ = param_path;
}

void RuntimeGraph::set_bin_path(const std::string &bin_path) {
  this->bin_path_ = bin_path;
}

const std::string &RuntimeGraph::param_path() const {
  return this->param_path_;
}

const std::string &RuntimeGraph::bin_path() const { return this->bin_path_; }

bool RuntimeGraph::Init() {
  if (this->param_path_.empty() || this->bin_path_.empty()) {
    LOG(ERROR) << "The param file path or bin file path is empty";
    return false;
  }

  // 加载pnnx格式的计算图
  this->graph_ = std::make_unique<pnnx::Graph>();
  int load_result = this->graph_->load(param_path_, bin_path_);
  if (load_result != 0) {
    LOG(ERROR) << "Load param file path and bin file path error: "
               << param_path_ << " " << bin_path_;
    return false;
  }

  // 读取pnnx计算图中的节点
  std::vector<pnnx::Operator *> pnnx_ops = this->graph_->ops;
  if (pnnx_ops.empty()) {
    LOG(ERROR) << "Can not read the pnnx runtime graph operators";
    return false;
  }

  // 依据pnnx计算图节点构造TinyInfer计算图节点
  this->ops_.clear();
  for (const pnnx::Operator *pnnx_op : pnnx_ops) {
    if (!pnnx_op) {
      LOG(ERROR) << "Meet the empty operator";
      continue;
    } else {
      srunop op = std::make_shared<RuntimeOp>();
      // 初始化节点名称、类型
      op->name = pnnx_op->name;
      op->type = pnnx_op->type;

      // 初始化节点的输入操作数
      // 包括操作数名称、维度、值类型
      const std::vector<pnnx::Operand *> &inputs = pnnx_op->inputs;
      if (!inputs.empty()) {
        InitOpInputs(inputs, op);
      }

      // 初始化节点的输出操作数
      // 注意：这里仅记录了当前节点的后继节点名称！
      const std::vector<pnnx::Operand *> &outputs = pnnx_op->outputs;
      if (!outputs.empty()) {
        InitOpOutputs(outputs, op);
      }

      // 初始化节点的参数
      const std::map<std::string, pnnx::Parameter> &params = pnnx_op->params;
      if (!params.empty()) {
        InitOpParams(params, op);
      }

      // 初始化节点的权重
      const std::map<std::string, pnnx::Attribute> &attrs = pnnx_op->attrs;
      if (!attrs.empty()) {
        InitOpAttrs(attrs, op);
      }

      // 保存构造好的节点
      this->ops_.push_back(op);
    }
  }
  // 初始化完毕，更新计算图状态为待构建
  graph_state_ = GraphState::NeedBuild;

  return true;
}

void RuntimeGraph::Build(const std::string &input_name,
                         const std::string &output_name) {
  if (graph_state_ == GraphState::NeedInit) {
    bool init_graph = Init();
    CHECK(init_graph == true) << "Init graph failed!";
  }

  CHECK(graph_state_ >= GraphState::NeedBuild)
      << "Graph status error, current state is " << int(graph_state_);

  if (graph_state_ == GraphState::Complete) {
    return;
  }

  CHECK(!this->ops_.empty()) << "Graph operators are empty, may be no init";

  // 遍历每个计算节点，保存其后继节点
  for (const auto &cur_op : this->ops_) {
    // 获取当前节点的后继节点（仅有名称）
    auto &out_ops = cur_op->out_ops;
    for (const auto &next_op : this->ops_) {
      if (next_op == cur_op) {
        continue;
      }
      // 根据名称找到后继节点并保存
      if (out_ops.find(next_op->name) != out_ops.end()) {
        out_ops.at(next_op->name) = next_op;
      }
    }
  }

  // 构造节点的计算Kernel
  // 注意：单独保存输入、输出节点，不用构造Kernel
  this->input_ops.clear();
  this->output_ops.clear();

  for (const auto &op : this->ops_) {
    if (op->type == "pnnx.Input") {
      this->input_ops.insert({op->name, op});
    } else if (op->type == "pnnx.Output") {
      this->output_ops.insert({op->name, op});
    } else {
      skernel kernel = RuntimeGraph::CreateKernel(op);
      CHECK(kernel != nullptr) << "Kernel create failed!";
      op->kernel = kernel;
      kernel->set_runtime_op(op);
    }
  }

  // 初始化节点的输入、输出空间
  RuntimeOperatorUtils::InitOpsInput(this->ops_);
  RuntimeOperatorUtils::InitOpsOutput(graph_->ops, this->ops_);

  graph_state_ = GraphState::Complete;
  input_name_ = input_name;
  output_name_ = output_name;

  // 销毁pnnx计算图
  if (graph_ != nullptr) {
    graph_.reset();
    graph_ = nullptr;
  }
}

std::vector<sftensor> RuntimeGraph::Forward(const std::vector<sftensor> &inputs,
                                            bool debug) {
  // 检查计算图是否构建完毕
  if (graph_state_ < GraphState::Complete) {
    LOG(FATAL) << "Graph need be build!";
  }
  CHECK(graph_state_ == GraphState::Complete)
      << "Graph status error, current state is " << int(graph_state_);

  // 找到计算图的输入节点
  srunop input_op;
  if (input_ops.find(input_name_) == input_ops.end()) {
    LOG(FATAL) << "Can not find the input operator: " << input_name_;
  } else {
    input_op = input_ops.at(input_name_);
  }

  // 找到计算图的输出节点
  srunop output_op;
  if (output_ops.find(output_name_) == output_ops.end()) {
    LOG(FATAL) << "Can not find the output operator: " << output_name_;
  } else {
    output_op = output_ops.at(output_name_);
  }

  // 计算图执行的辅助队列——基于BFS执行计算图！
  std::deque<srunop> ops_que;
  // 输入节点入队
  ops_que.push_back(input_op);

  std::unordered_map<std::string, double> run_dur_infos; // 统计运行时间

  if (debug) {
    LOG(INFO) << "Batch Size:" << inputs.size();
    for (int b = 0; b < inputs.size(); ++b) {
      LOG(INFO) << "Input Channels: " << inputs.at(b)->channels()
                << " Rows: " << inputs.at(b)->rows()
                << " Cols: " << inputs.at(b)->cols();
    }
    LOG(INFO) << "Inference starting...";
    LOG(INFO) << "--------------------------------------------------"
              << "\n";
  }

  while (!ops_que.empty()) {
    // 取出队头节点
    srunop cur_op = ops_que.front();
    ops_que.pop_front();

    // 当前节点为空或为输出节点，推理结束
    if (!cur_op || cur_op == output_op) {
      if (debug) {
        LOG(INFO) << "Inference ended...";
      }
      break;
    }
    // 当前节点为输入节点，其输出Tensor等于输入inputs
    else if (cur_op == input_op) {
      ProbeNextOp(cur_op, ops_que, inputs);
    }
    // 当前节点不是输入节点，首先检测是否就绪，若就绪则执行
    else {

      const auto &start = std::chrono::steady_clock::now();
      // 执行当前节点
      InferStatus status = cur_op->kernel->Forward();

      CHECK(status == InferStatus::InferSuccess)
          << cur_op->kernel->kernel_name()
          << " kernel forward failed, error code: " << int(status);

      // 统计相同类型算子累计执行时间
      if (debug) {
        const double dura =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - start)
                .count();
        if (run_dur_infos.find(cur_op->type) == run_dur_infos.end()) {
          run_dur_infos.insert({cur_op->type, dura});
        } else {
          run_dur_infos.at(cur_op->type) += dura;
        }
      }

      const auto copy_start = std::chrono::steady_clock::now();
      // 将当前节点的输出Tensor传递给后继节点
      ProbeNextOp(cur_op, ops_que, cur_op->out_oprand->data);

      // 统计不同层间传递Tensor的累计时间
      if (debug) {
        const double dura =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - copy_start)
                .count();
        if (run_dur_infos.find("Copy") == run_dur_infos.end()) {
          run_dur_infos.insert({"Copy", dura});
        } else {
          run_dur_infos.at("Copy") += dura;
        }
      }
    }
  }

  // 所有节点执行完毕，全部置为未就绪状态
  for (const auto &op : this->ops_) {
    op->meet_num = 0;
  }

  if (debug) {
    LOG(INFO) << "Inference Information, Time Cost:";
    double dura_sum = 0.;
    for (const auto &[type, dura] : run_dur_infos) {
      LOG(INFO) << "OP type: " << type << " duration: " << dura << " s";
      dura_sum += dura;
    }
    LOG(INFO) << "All time cost: " << dura_sum << " s";
  }

  // 检查输出节点的输入数目是否为1
  CHECK(output_op->in_oprands.size() == 1)
      << "TinyInfer only supports one input oprand to the output operator!";

  // 输出节点的输入就是整个计算图的输出
  const auto &out_oprand = output_op->in_oprands.begin()->second;

  return out_oprand->data;
}

skernel RuntimeGraph::CreateKernel(const srunop &op) {
  CHECK(op != nullptr) << "Operator is empty!";
  const auto &kernel = KernelRegister::CreateKernel(op);
  CHECK(kernel != nullptr) << "Kernel init failed " << op->type;
  return kernel;
}

void RuntimeGraph::InitOpInputs(const std::vector<pnnx::Operand *> &inputs,
                                const srunop &op) {
  // 遍历所有来源的输入操作数
  for (const pnnx::Operand *input : inputs) {
    if (!input) {
      continue;
    }
    const pnnx::Operator *producer =
        input->producer; // 获取输入操作数的生产节点
    srunoprand oprand = std::make_shared<RuntimeOprand>();
    // 初始化输入操作数的名称、维度
    // 注意：输入操作数名称是其生产节点名称
    oprand->name = producer->name;
    oprand->shape = input->shape;

    // 初始化输入的值类型
    switch (input->type) {
    case 1: {
      oprand->type = RuntimeDataType::TypeFloat32;
      break;
    }
    case 0: {
      oprand->type = RuntimeDataType::TypeUnknown;
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported input operand type: " << input->type;
    }
    }

    op->in_oprands_seq.push_back(oprand);
    op->in_oprands.insert({oprand->name, oprand});
  }
}

void RuntimeGraph::InitOpOutputs(const std::vector<pnnx::Operand *> &outputs,
                                 const srunop &op) {
  for (const pnnx::Operand *output : outputs) {
    if (!output) {
      continue;
    }
    // 获取输出操作数的消费（后继）节点
    const auto &consumers = output->consumers;
    // 保存当前节点的后继节点名称
    for (const auto &consumer : consumers) {
      op->out_ops.insert({consumer->name, nullptr});
    }
  }
}

void RuntimeGraph::InitOpParams(
    const std::map<std::string, pnnx::Parameter> &params, const srunop &op) {
  // 遍历每一个参数
  for (const auto &[name, pnnx_param] : params) {
    // 构造不同类型的参数
    const int type = pnnx_param.type;
    switch (type) {
    case int(RuntimeParamType::ParamUnknown): {
      RuntimeParam *param = new RuntimeParam;
      op->params.insert({name, param});
      break;
    }

    case int(RuntimeParamType::ParamBool): {
      RuntimeParamBool *param = new RuntimeParamBool;
      param->value = pnnx_param.b;
      op->params.insert({name, param});
      break;
    }

    case int(RuntimeParamType::ParamInt): {
      RuntimeParamInt *param = new RuntimeParamInt;
      param->value = pnnx_param.i;
      op->params.insert({name, param});
      break;
    }

    case int(RuntimeParamType::ParamFloat): {
      RuntimeParamFloat *param = new RuntimeParamFloat;
      param->value = pnnx_param.f;
      op->params.insert({name, param});
      break;
    }

    case int(RuntimeParamType::ParamStr): {
      RuntimeParamStr *param = new RuntimeParamStr;
      param->value = pnnx_param.s;
      op->params.insert({name, param});
      break;
    }

    case int(RuntimeParamType::ParamIntArray): {
      RuntimeParamIntArr *param = new RuntimeParamIntArr;
      param->value = pnnx_param.ai;
      op->params.insert({name, param});
      break;
    }

    case int(RuntimeParamType::ParamFloatArray): {
      RuntimeParamFloatArr *param = new RuntimeParamFloatArr;
      param->value = pnnx_param.af;
      op->params.insert({name, param});
      break;
    }

    case int(RuntimeParamType::ParamStrArray): {
      RuntimeParamStrArr *param = new RuntimeParamStrArr;
      param->value = pnnx_param.as;
      op->params.insert({name, param});
      break;
    }

    default: {
      LOG(FATAL) << "Unsupported parameter type";
    }
    }
  }
}

void RuntimeGraph::InitOpAttrs(
    const std::map<std::string, pnnx::Attribute> &attrs, const srunop &op) {
  // 遍历每一个权重
  for (const auto &[name, pnnx_attr] : attrs) {
    switch (pnnx_attr.type) {
    case 1: {
      srunattr attr = std::make_shared<RuntimeAttr>();
      attr->type = RuntimeDataType::TypeFloat32;
      attr->shape = pnnx_attr.shape;
      attr->weight_data = pnnx_attr.data;
      op->attrs.insert({name, attr});
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported attribute type";
    }
    }
  }
}

bool RuntimeGraph::CheckOpReady(const srunop &op) {
  CHECK(op != nullptr);
  CHECK(op->meet_num <= op->in_oprands.size());

  // 注意：当前节点的被访问次数等于输入数目，说明其所有前驱节点都已被执行，节点就绪，可被执行
  if (op->meet_num == op->in_oprands.size()) {
    return true;
  } else {
    return false;
  }
}

void RuntimeGraph::ProbeNextOp(const srunop &cur_op,
                               std::deque<srunop> &ops_que,
                               const std::vector<sftensor> &outputs) {
  // 遍历当前节点的后继节点
  const auto &next_ops = cur_op->out_ops;
  for (const auto &[next_name, next_op] : next_ops) {
    // 取出后继节点的输入操作数
    const auto &next_in_oprands = next_op->in_oprands;
    // 检查后继节点的输入操作数是否来自当前节点
    if (next_in_oprands.find(cur_op->name) != next_in_oprands.end()) {
      // 将当前节点的输出Tensor拷贝给后继节点（拷贝指针，两个指针指向同一Tensor对象）
      std::vector<sftensor> &next_in_data =
          next_in_oprands.at(cur_op->name)->data;
      for (int b = 0; b < next_in_data.size(); ++b) {
        next_in_data.at(b) = outputs.at(b);
      }

      // 后继节点的被访问次数加1
      next_op->meet_num += 1;

      // 检测后继节点是否已经就绪，若就绪，则加入队列
      if (CheckOpReady(next_op)) {
        ops_que.push_back(next_op);
      }
    }
  }
}

} // namespace TinyInfer
