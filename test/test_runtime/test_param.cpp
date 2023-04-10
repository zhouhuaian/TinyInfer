#include <gtest/gtest.h>
#include "runtime/runtime_parameter.hpp"

using namespace TinyInfer;

TEST(test_runtime, runtime_param1) {
  RuntimeParameter* param = new RuntimeParameterInt;
  ASSERT_EQ(param->type, RuntimeParameterType::ParamInt);
}

TEST(test_runtime, runtime_param2) {
  RuntimeParameter* param = new RuntimeParameterInt;
  ASSERT_EQ(param->type, RuntimeParameterType::ParamInt);
  ASSERT_EQ(dynamic_cast<RuntimeParameterFloat*>(param), nullptr);
  ASSERT_NE(dynamic_cast<RuntimeParameterInt*>(param), nullptr);
}