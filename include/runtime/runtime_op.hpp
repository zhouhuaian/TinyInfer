#ifndef TINY_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
#define TINY_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_

#include "kernel/abstract/kernel.hpp"
#include "runtime/ir.h"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_param.hpp"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace TinyInfer {

class Kernel;

// 计算图中的计算节点
struct RuntimeOp {
  virtual ~RuntimeOp();

  std::string name;               // 计算节点的名称
  std::string type;               // 计算节点的类型
  std::shared_ptr<Kernel> kernel; // 节点对应的计算Kernel
  int32_t meet_num = 0;           // 当前节点被前驱节点访问的次数

  std::vector<srunoprand>
      in_oprands_seq; // 节点的输入操作数（一个节点可能有多个来源的输入）
  std::unordered_map<std::string, srunoprand>
      in_oprands; // 节点的输入操作数（名称和操作数的映射）

  srunoprand out_oprand; // 节点的输出操作数（一个节点最多有一个输出）
  std::unordered_map<std::string, std::shared_ptr<RuntimeOp>>
      out_ops; // 后继（输出）节点

  std::unordered_map<std::string, RuntimeParam *> params; // 节点参数
  std::unordered_map<std::string, srunattr> attrs;        // 节点权重
};

using srunop = std::shared_ptr<RuntimeOp>;

class RuntimeOperatorUtils {
public:
  /**
   * 初始化节点的输入空间
   * @param ops 计算图节点
   */
  static void InitOpsInput(const std::vector<srunop> &ops);

  /**
   * 初始化节点的输出空间
   * @param pnnx_ops pnnx格式计算图节点
   * @param ops 计算图节点
   */
  static void InitOpsOutput(const std::vector<pnnx::Operator *> &pnnx_ops,
                            const std::vector<srunop> &ops);
};

} // namespace TinyInfer

#endif // TINY_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
