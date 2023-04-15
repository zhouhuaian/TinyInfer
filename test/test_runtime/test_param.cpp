#include <gtest/gtest.h>
#include "runtime/runtime_param.hpp"

using namespace TinyInfer;

TEST(test_runtime, runtime_param1) {
  RuntimeParam* param = new RuntimeParamInt;
  ASSERT_EQ(param->type, RuntimeParamType::ParamInt);
}

TEST(test_runtime, runtime_param2) {
  RuntimeParam* param = new RuntimeParamInt;
  ASSERT_EQ(param->type, RuntimeParamType::ParamInt);
  ASSERT_EQ(dynamic_cast<RuntimeParamFloat*>(param), nullptr);
  ASSERT_NE(dynamic_cast<RuntimeParamInt*>(param), nullptr);
}