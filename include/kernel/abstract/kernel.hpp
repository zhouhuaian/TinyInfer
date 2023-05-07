#ifndef TINY_INFER_SOURCE_KERNEL_KERNEL_HPP_
#define TINY_INFER_SOURCE_KERNEL_KERNEL_HPP_

#include "data/tensor.hpp"
#include "runtime/runtime_op.hpp"
#include "status_code.hpp"
#include <glog/logging.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace TinyInfer {

// 前置声明
class RuntimeOp;
using srunop = std::shared_ptr<RuntimeOp>;

// 计算图节点对应的Kernel——真正负责推理计算的类
class Kernel {
public:
  explicit Kernel(std::string name) : name_(name) {}

  virtual ~Kernel() = default;

  /**
   * Kernel的执行函数
   * @return 执行状态
   */
  virtual InferStatus Forward();

  /**
   * Kernel的执行函数
   * @param inputs Kernel的输入Tensor
   * @param outputs Kernel的输出Tensor
   * @return 执行状态
   */
  virtual InferStatus Forward(const std::vector<sftensor> &inputs,
                              std::vector<sftensor> &outputs);

  /**
   * 设置Kernel的权重
   */
  virtual void set_weights(const std::vector<sftensor> &weights);

  /**
   * 设置Kernel的权重
   */
  virtual void set_weights(const std::vector<float> &weights);

  /**
   * 返回Kernel的权重
   */
  virtual const std::vector<sftensor> &weights() const;

  /**
   * 设置Kernel的偏置
   */
  virtual void set_bias(const std::vector<sftensor> &bias);

  /**
   * 设置Kernel的偏置
   */
  virtual void set_bias(const std::vector<float> &bias);

  /**
   * 返回Kernel的偏置
   */
  virtual const std::vector<sftensor> &bias() const;

  /**
   * 返回Kernel的名称
   */
  virtual const std::string &kernel_name() const { return this->name_; }

  /**
   * 设置Kernel对应的计算节点
   * @param op 计算节点
   */
  void set_runtime_op(const srunop &op);

protected:
  std::string name_; // Kernel的名称——与计算节点类型名不一样

  std::weak_ptr<RuntimeOp> op_; // Kernel对应的计算节点——用weak_ptr避免循环引用
};

using skernel = std::shared_ptr<Kernel>;

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_KERNEL_HPP_
