#ifndef TINY_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
#define TINY_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_

#include <memory>
#include <vector>
#include <glog/logging.h>
#include "runtime_datatype.hpp"
#include "status_code.hpp"

namespace TinyInfer {

// 计算节点的权重
struct RuntimeAttr {
  
  std::vector<char> weight_data;  // 权重值（以char类型存储）
  std::vector<int> shape;         // 权重维度
  RuntimeDataType type = RuntimeDataType::TypeUnknown;  // 权重值类型

  /**
   * 获取权重值
   * @tparam T 权重值类型
   * @return 权重值数组
   */
  template <class T>
  std::vector<T> get(bool need_clear = true);

  /**
   * 清除权重
   */
  void clear();
};

using srunattr = std::shared_ptr<RuntimeAttr>;

template <class T>
std::vector<T> RuntimeAttr::get(bool need_clear) {
  CHECK(!weight_data.empty());
  CHECK(type != RuntimeDataType::TypeUnknown);
  
  std::vector<T> weights;
  switch (type) {
    // 目前只支持float32
    case RuntimeDataType::TypeFloat32: {
      // 检查模板参数是否为float类型
      const bool is_float = std::is_same<T, float>::value;
      CHECK_EQ(is_float, true);
      // 获取当前机器上float类型占用的字节数
      const uint32_t float_size = sizeof(float);
      // 判断weight_data中的字节数是float_size的整数倍
      CHECK_EQ(weight_data.size() % float_size, 0);
      // 按float_size大小读取（解释）权重值
      for (uint32_t i = 0; i < weight_data.size() / float_size; ++i) {
        float weight = *((float*)weight_data.data() + i);
        weights.push_back(weight);
      }
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported weight data type!";
    }
  }

  if (need_clear) {
    this->clear();
  }

  return weights;
}

}  // namespace TinyInfer

#endif  // TINY_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
