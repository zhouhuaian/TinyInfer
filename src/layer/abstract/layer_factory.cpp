#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include "status_code.hpp"
#include <glog/logging.h>

namespace TinyInfer {

LayerRegister::Registry& LayerRegister::CreateRegistry() {
  static Registry* LayerRegistry = new Registry();
  CHECK(LayerRegistry != nullptr) << "Global layer register created failed!";
  return *LayerRegistry;
}

void LayerRegister::RegisterCreator(const std::string& op_type,
                                      const Creator& creator) {
  CHECK(creator != nullptr);
  
  Registry& registry = CreateRegistry();  // 取出注册表

  CHECK(registry.find(op_type) == registry.end())
      << "Layer type: " << op_type << " has been registered!";
  
  registry.insert({op_type, creator});
}

slayer LayerRegister::CreateLayer(const srunop& op) {
  Registry& registry = CreateRegistry();

  const std::string& op_type = op->type;
  CHECK(registry.find(op_type) != registry.end())
      << "Can not find the layer type: " << op_type;
  
  const auto& creator = registry[op_type];
  CHECK(creator != nullptr) << "Layer creator is empty!";
  
  // 调用creator创建op对应的layer
  slayer layer;
  const auto& status = creator(op, layer);
  CHECK(status == ParseParamAttrStatus::ParamAttrParseSuccess)
      << "Create the layer: " << op_type
      << " failed, error code: " << int(status);
  return layer;
}

}  // namespace TinyInfer
