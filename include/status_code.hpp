#ifndef TINY_INFER_INCLUDE_COMMON_HPP_
#define TINY_INFER_INCLUDE_COMMON_HPP_

namespace TinyInfer {

// 计算节点参数值类型
enum class RuntimeParamType {
  ParamUnknown = 0,

  ParamBool = 1,
  ParamInt = 2,
  ParamFloat = 3,
  ParamStr = 4,

  ParamIntArray = 5,
  ParamFloatArray = 6,
  ParamStrArray = 7,
};

// 推理状态
enum class InferStatus {
  InferUnknown = -1,
  InferSuccess = 0,

  InferFailedInputEmpty = 1,      // 输入Tensor为空
  InferFailedBatchMatchError = 2, // 输入输出Tensor数目不匹配
};

enum class ParseParamAttrStatus {
  ParamAttrParseSuccess = 0,

  OpEmpty = 1,
  ParamMissingStride = 2,
  ParamMissingPadding = 3,
  ParamMissingKernelSize = 4,
  ParamMissingBias = 5,
  ParamMissingInChannels = 6,
  ParamMissingOutChannels = 7,
  ParamMissingDim = 8,
  ParamMissingExpr = 9,
  ParamMissingOutHW = 10,
  ParamMissingGroups = 11,
  ParamMissingDilation = 12,
  ParamMissingPaddingMode = 13,

  AttrMissingBias = 14,
  AttrMissingWeight = 15,
  AttrMissingOutFeatures = 16,
};

} // namespace TinyInfer

#endif // TINY_INFER_INCLUDE_COMMON_HPP_
