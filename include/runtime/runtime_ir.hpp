#ifndef TINY_INFER_RUNTIM_GRAPH_HPP_
#define TINY_INFER_RUNTIM_GRAPH_HPP_

#include "ir.h"
#include "layer/abstract/layer.hpp"
#include "runtime/runtime_operand.hpp"
#include "runtime_op.hpp"
#include <glog/logging.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace TinyInfer {

// 计算图，由计算节点和节点间的操作数流构成
class RuntimeGraph {
public:
  /**
   * 初始化计算图
   * @param param_path 计算图结构文件
   * @param bin_path 计算图的权重文件
   */
  RuntimeGraph(std::string param_path, std::string bin_path);

  /**
   * 构建计算图
   * @param input_name 输入节点名称
   * @param output_name  输出节点名称
   */
  void Build(const std::string &input_name, const std::string &output_name);

  /**
   * 设置结构文件路径
   */
  void set_param_path(const std::string &param_path);

  /**
   * 设置权重文件路径
   */
  void set_bin_path(const std::string &bin_path);

  /**
   * 返回结构文件路径
   */
  const std::string &param_path() const;

  /**
   * 返回权重文件路径
   */
  const std::string &bin_path() const;

  /**
   * 执行计算图推理
   * @param inputs 计算图的输入Tensor（一个批次）
   * @param debug 是否调试，若调试，则会输出中间信息
   * @return 计算图的输出Tensor（一个批次）
   */
  std::vector<sftensor> Forward(const std::vector<sftensor> &inputs,
                                bool debug = false);

private:
  /**
   * 初始化计算图
   * @return 是否初始化成功
   */
  bool Init();

  /**
   * 初始化计算图节点的输入操作数
   * @param inputs pnnx计算图节点的输入操作数
   * @param op 计算图节点
   */
  static void InitOpInputs(const std::vector<pnnx::Operand *> &inputs,
                           const srunop &op);

  /**
   * 初始化计算图节点的输出操作数
   * @param outputs pnnx计算图节点的输出操作数
   * @param op 计算图节点
   */
  static void InitOpOutputs(const std::vector<pnnx::Operand *> &outputs,
                            const srunop &op);

  /**
   * 初始化计算图节点的参数
   * @param params pnnx计算图节点的参数
   * @param op 计算图节点
   */
  static void InitOpParams(const std::map<std::string, pnnx::Parameter> &params,
                           const srunop &op);

  /**
   * 初始化计算图节点的权重
   * @param attrs pnnx计算图节点的权重
   * @param op 计算图节点
   */
  static void InitOpAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                          const srunop &op);

  /**
   * 创建节点的计算Layer
   * @param op 计算图节点
   * @return 创建成功的计算Layer
   */
  static slayer CreateLayer(const srunop &op);

  /**
   * 检查当前节点是否就绪
   * @param op 待检查的节点
   */
  static bool CheckOpReady(const srunop &op);

  /**
   * 向后继节点传递操作数，并将就绪的节点加入执行队列
   * @param cur_op 当前节点
   * @param ops_que 计算图节点的执行队列
   * @param outputs 当前节点的输出Tensor，会被传递给后继节点
   */
  static void ProbeNextOp(const srunop &cur_op, std::deque<srunop> &ops_que,
                          const std::vector<sftensor> &outputs);

private:
  // 计算图状态类型
  enum class GraphState {
    NeedInit = -2,  // 待初始化
    NeedBuild = -1, // 待构建
    Complete = 0,   // 构建完毕
  };

  GraphState graph_state_;  // 计算图状态
  std::string param_path_;  // 计算图结构文件
  std::string bin_path_;    // 计算图权重文件
  std::string input_name_;  // 输入节点名称
  std::string output_name_; // 输出节点名称

  std::unique_ptr<pnnx::Graph> graph_;                // pnnx格式的计算图
  std::vector<srunop> ops_;                           // 计算图节点
  std::unordered_map<std::string, srunop> input_ops;  // 输入节点
  std::unordered_map<std::string, srunop> output_ops; // 输出节点
};

} // namespace TinyInfer

#endif // TINY_INFER_RUNTIM_GRAPH_HPP_