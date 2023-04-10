#ifndef TINY_INFER_INCLUDE_PARSER_RUNTIME_PARAMETER_HPP_
#define TINY_INFER_INCLUDE_PARSER_RUNTIME_PARAMETER_HPP_

#include <string>
#include <vector>
#include "status_code.hpp"

namespace TinyInfer {

// 计算节点的参数
struct RuntimeParameter {
  explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::ParamUnknown) : type(type) {}

  virtual ~RuntimeParameter() = default;

  RuntimeParameterType type;  // 参数值类型
};

// int类型参数
struct RuntimeParameterInt : public RuntimeParameter {
  RuntimeParameterInt() : RuntimeParameter(RuntimeParameterType::ParamInt) {}
  
  int value = 0;
};

// float类型参数
struct RuntimeParameterFloat : public RuntimeParameter {
  RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::ParamFloat) {}

  float value = 0.f;
};

// string类型参数
struct RuntimeParameterString : public RuntimeParameter {
  RuntimeParameterString() : RuntimeParameter(RuntimeParameterType::ParamStr) {}

  std::string value;
};

// int array类型参数
struct RuntimeParameterIntArray : public RuntimeParameter {
  RuntimeParameterIntArray() : RuntimeParameter(RuntimeParameterType::ParamIntArray) {}

  std::vector<int> value;
};

// float array类型参数
struct RuntimeParameterFloatArray : public RuntimeParameter {
  RuntimeParameterFloatArray() : RuntimeParameter(RuntimeParameterType::ParamFloatArray) {}

  std::vector<float> value;
};

// string array类型参数
struct RuntimeParameterStringArray : public RuntimeParameter {
  RuntimeParameterStringArray() : RuntimeParameter(RuntimeParameterType::ParamStrArray) {}

  std::vector<std::string> value;
};

// bool类型参数
struct RuntimeParameterBool : public RuntimeParameter {
  RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::ParamBool) {}

  bool value = false;
};

}  // namespace TinyInfer

#endif  // TINY_INFER_INCLUDE_PARSER_RUNTIME_PARAMETER_HPP_
