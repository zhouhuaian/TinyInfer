#ifndef TINY_INFER_INCLUDE_PARSER_RUNTIME_OPRAND_HPP_
#define TINY_INFER_INCLUDE_PARSER_RUNTIME_OPRAND_HPP_

#include "data/tensor.hpp"
#include "runtime_datatype.hpp"
#include "status_code.hpp"
#include <memory>
#include <string>
#include <vector>

namespace TinyInfer {

// 计算图中的操作数
struct RuntimeOprand {
  std::string name;           // 操作数名称
  std::vector<int32_t> shape; // 操作数维度
  std::vector<sftensor>
      data; // 存储操作数——data以[tensor1,tensor2,...]存放一个批次
  RuntimeDataType type = RuntimeDataType::TypeUnknown; // 操作数值类型
};

using srunoprand = std::shared_ptr<RuntimeOprand>;

} // namespace TinyInfer

#endif // TINY_INFER_INCLUDE_PARSER_RUNTIME_OPRAND_HPP_
