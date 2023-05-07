#include "kernel/abstract/kernel_factory.hpp"
#include <gtest/gtest.h>

using namespace TinyInfer;

ParseParamAttrStatus TestCreateKernel(const srunop &op, skernel &kernel) {
  kernel = std::make_shared<Kernel>("test3");
  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

TEST(test_kernel_factory, init) {
  const auto &reg1 = KernelRegister::CreateRegistry();
  const auto &reg2 = KernelRegister::CreateRegistry();
  ASSERT_EQ(reg1, reg2);
}

TEST(test_kernel_factory, registerer) {
  const uint32_t size1 = KernelRegister::CreateRegistry().size();
  KernelRegister::RegisterCreator("test1", TestCreateKernel);
  const uint32_t size2 = KernelRegister::CreateRegistry().size();
  KernelRegister::RegisterCreator("test2", TestCreateKernel);
  const uint32_t size3 = KernelRegister::CreateRegistry().size();

  ASSERT_EQ(size2 - size1, 1);
  ASSERT_EQ(size3 - size1, 2);
}

TEST(test_kernel_factory, create) {
  KernelRegister::RegisterCreator("test3", TestCreateKernel);
  srunop op = std::make_shared<RuntimeOp>();
  op->type = "test3";
  skernel kernel = KernelRegister::CreateKernel(op);
  ASSERT_EQ(kernel->kernel_name(), "test3");
}