#ifndef TINY_INFER_INCLUDE_RUNTIME_RUNTIME_DATATYPE_HPP_
#define TINY_INFER_INCLUDE_RUNTIME_RUNTIME_DATATYPE_HPP_

namespace TinyInfer {

// 计算图中的数值类型
enum class RuntimeDataType {
  TypeUnknown = 0,
  TypeFloat32 = 1,
  TypeFloat64 = 2,
  TypeFloat16 = 3,
  TypeInt32 = 4,
  TypeInt64 = 5,
  TypeInt16 = 6,
  TypeInt8 = 7,
  TypeUInt8 = 8,
};

} // namespace TinyInfer

#endif // TINY_INFER_INCLUDE_RUNTIME_RUNTIME_DATATYPE_HPP_
