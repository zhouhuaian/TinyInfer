#ifndef TINY_INFER_SOURCE_LAYER_RELU_HPP_
#define TINY_INFER_SOURCE_LAYER_RELU_HPP_

#include "layer/abstract/no_attr_layer.hpp"

namespace TinyInfer {

class ReLU : public NoAttrLayer {
public:
    explicit ReLU();

    /**
     * 计算函数——实现ReLU layer的计算
     * @param inputs 输入Tensor
     * @param outputs 输出Tensor
     */
    InferStatus Forward(const std::vector<sftensor>& inputs, std::vector<sftensor>& outputs) override;

    /**
     * 创建ReLU layer
     * @param op 计算图节点
     * @param relu 创建的ReLU layer
     */
    static ParseParamAttrStatus GetInstance(const srunop& op, slayer& relu);
};

}  // namespace TinyInfer

#endif  // TINY_INFER_SOURCE_LAYER_RELU_HPP_
