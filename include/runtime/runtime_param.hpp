#ifndef TINY_INFER_INCLUDE_PARSER_RUNTIME_PARAM_HPP_
#define TINY_INFER_INCLUDE_PARSER_RUNTIME_PARAM_HPP_

#include <string>
#include <vector>
#include "status_code.hpp"

namespace TinyInfer {

// 计算节点的参数
struct RuntimeParam {
  explicit RuntimeParam(RuntimeParamType type = RuntimeParamType::ParamUnknown) : type(type) {}

  virtual ~RuntimeParam() = default;

  RuntimeParamType type;  // 参数值类型
};

// int类型参数
struct RuntimeParamInt : public RuntimeParam {
  RuntimeParamInt() : RuntimeParam(RuntimeParamType::ParamInt) {}
  
  int value = 0;
};

// float类型参数
struct RuntimeParamFloat : public RuntimeParam {
  RuntimeParamFloat() : RuntimeParam(RuntimeParamType::ParamFloat) {}

  float value = 0.f;
};

// string类型参数
struct RuntimeParamStr : public RuntimeParam {
  RuntimeParamStr() : RuntimeParam(RuntimeParamType::ParamStr) {}

  std::string value;
};

// int array类型参数
struct RuntimeParamIntArr : public RuntimeParam {
  RuntimeParamIntArr() : RuntimeParam(RuntimeParamType::ParamIntArray) {}

  std::vector<int> value;
};

// float array类型参数
struct RuntimeParamFloatArr : public RuntimeParam {
  RuntimeParamFloatArr() : RuntimeParam(RuntimeParamType::ParamFloatArray) {}

  std::vector<float> value;
};

// string array类型参数
struct RuntimeParamStrArr : public RuntimeParam {
  RuntimeParamStrArr() : RuntimeParam(RuntimeParamType::ParamStrArray) {}

  std::vector<std::string> value;
};

// bool类型参数
struct RuntimeParamBool : public RuntimeParam {
  RuntimeParamBool() : RuntimeParam(RuntimeParamType::ParamBool) {}

  bool value = false;
};

}  // namespace TinyInfer

#endif  // TINY_INFER_INCLUDE_PARSER_RUNTIME_PARAM_HPP_
