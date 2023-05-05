#ifndef TINY_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#define TINY_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_

#include "layer.hpp"
#include "runtime/runtime_op.hpp"
#include <memory>
#include <string>
#include <unordered_map>

namespace TinyInfer {

class LayerRegister {
public:
  // Layer的创建函数类型
  using Creator = ParseParamAttrStatus (*)(const srunop &op, slayer &layer);
  // 注册表类型
  using Registry = std::unordered_map<std::string, Creator>;

  /**
   * 创建全局注册表
   * @return 注册表
   */
  static Registry &CreateRegistry();

  /**
   * 注册Layer的创建函数到注册表中
   * @param op_type 计算节点类型
   * @param creator Layer的创建函数
   */
  static void RegisterCreator(const std::string &op_type,
                              const Creator &creator);

  /**
   * 创建Layer
   * @param op 待创建Layer的计算节点
   * @return 创建的Layer
   */
  static slayer CreateLayer(const srunop &op);
};

class LayerRegisterWrapper {
public:
  LayerRegisterWrapper(const std::string &op_type,
                       const LayerRegister::Creator &creator) {
    LayerRegister::RegisterCreator(op_type, creator);
  }
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
