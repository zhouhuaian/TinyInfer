#ifndef TINY_INFER_SOURCE_KERNEL_KERNEL_FACTORY_HPP_
#define TINY_INFER_SOURCE_KERNEL_KERNEL_FACTORY_HPP_

#include "kernel.hpp"
#include "runtime/runtime_op.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace TinyInfer {

class KernelRegister {
public:
  // Kernel的创建函数类型
  using Creator = ParseParamAttrStatus (*)(const srunop &op, skernel &kernel);
  // 注册表类型
  using Registry = std::unordered_map<std::string, Creator>;

  /**
   * 创建全局注册表
   * @return 注册表
   */
  static Registry &CreateRegistry();

  /**
   * 注册Kernel的创建函数到注册表中
   * @param op_type 计算节点类型
   * @param creator Kernel的创建函数
   */
  static void RegisterCreator(const std::string &op_type,
                              const Creator &creator);

  /**
   * 创建Kernel
   * @param op 待创建Kernel的计算节点
   * @return 创建的Kernel
   */
  static skernel CreateKernel(const srunop &op);
};

class KernelRegisterWrapper {
public:
  KernelRegisterWrapper(const std::string &op_type,
                        const KernelRegister::Creator &creator) {
    KernelRegister::RegisterCreator(op_type, creator);
  }
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_KERNEL_FACTORY_HPP_
