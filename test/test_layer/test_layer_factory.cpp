#include <gtest/gtest.h>
#include "layer/abstract/layer_factory.hpp"

using namespace TinyInfer;

ParseParamAttrStatus TestCreateLayer(const srunop& op, slayer& layer) {
  layer = std::make_shared<Layer>("test3");
  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

TEST(test_layer_factory, init) {
  const auto& reg1 = LayerRegister::CreateRegistry();
  const auto& reg2 = LayerRegister::CreateRegistry();
  ASSERT_EQ(reg1, reg2);
}

TEST(test_layer_factory, registerer) {
  const uint32_t size1 = LayerRegister::CreateRegistry().size();
  LayerRegister::RegisterCreator("test1", TestCreateLayer);
  const uint32_t size2 = LayerRegister::CreateRegistry().size();
  LayerRegister::RegisterCreator("test2", TestCreateLayer);
  const uint32_t size3 = LayerRegister::CreateRegistry().size();

  ASSERT_EQ(size2 - size1, 1);
  ASSERT_EQ(size3 - size1, 2);
}

TEST(test_layer_factory, create) {
  LayerRegister::RegisterCreator("test3", TestCreateLayer);
  srunop op = std::make_shared<RuntimeOperator>();
  op->type = "test3";
  slayer layer = LayerRegister::CreateLayer(op);
  ASSERT_EQ(layer->layer_name(), "test3");
}