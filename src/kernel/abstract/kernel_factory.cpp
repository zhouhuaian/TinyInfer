#include "kernel/abstract/kernel_factory.hpp"
#include "runtime/runtime_graph.hpp"
#include "status_code.hpp"
#include <glog/logging.h>

namespace TinyInfer {

KernelRegister::Registry &KernelRegister::CreateRegistry() {
  static Registry *KernelRegistry = new Registry();
  CHECK(KernelRegistry != nullptr) << "Global kernel register created failed!";
  return *KernelRegistry;
}

void KernelRegister::RegisterCreator(const std::string &op_type,
                                     const Creator &creator) {
  CHECK(creator != nullptr);

  Registry &registry = CreateRegistry(); // 取出注册表

  CHECK(registry.find(op_type) == registry.end())
      << "Kernel type: " << op_type << " has been registered!";

  registry.insert({op_type, creator});
}

skernel KernelRegister::CreateKernel(const srunop &op) {
  Registry &registry = CreateRegistry();

  const std::string &op_type = op->type;
  CHECK(registry.find(op_type) != registry.end())
      << "Can not find the kernel type: " << op_type;

  const auto &creator = registry[op_type];
  CHECK(creator != nullptr) << "Kernel creator is empty!";

  // 调用creator创建op对应的kernel
  skernel kernel;
  const auto &status = creator(op, kernel);
  CHECK(status == ParseParamAttrStatus::ParamAttrParseSuccess)
      << "Create the kernel: " << op_type
      << " failed, error code: " << int(status);
  return kernel;
}

} // namespace TinyInfer
