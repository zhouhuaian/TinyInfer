#ifndef TINY_INFER_SOURCE_LAYER_LAYER_HPP_
#define TINY_INFER_SOURCE_LAYER_LAYER_HPP_

#include <glog/logging.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "data/tensor.hpp"
#include "runtime/runtime_op.hpp"
#include "status_code.hpp"

namespace TinyInfer {

// 前置声明
class RuntimeOperator;
using srunop = std::shared_ptr<RuntimeOperator>;

// 计算图节点对应的Layer——真正负责推理计算的类
class Layer {
public:
    explicit Layer(std::string name) : name_(name) {}

    virtual ~Layer() = default;

    /**
     * Layer的执行函数
     * @return 执行状态
     */
    virtual InferStatus Forward();

    /**
     * Layer的执行函数
     * @param inputs Layer的输入Tensor
     * @param outputs Layer的输出Tensor
     * @return 执行状态
     */
    virtual InferStatus Forward(const std::vector<sftensor>& inputs, std::vector<sftensor>& outputs);

    /**
     * 设置Layer的权重
     */
    virtual void set_weights(const std::vector<sftensor>& weights);

    /**
     * 设置Layer的权重
     */
    virtual void set_weights(const std::vector<float>& weights);

    /**
     * 返回Layer的权重
     */
    virtual const std::vector<sftensor>& weights() const;

    /**
     * 设置Layer的偏置
     */
    virtual void set_bias(const std::vector<sftensor>& bias);

    /**
     * 设置Layer的偏置
     */
    virtual void set_bias(const std::vector<float>& bias);

    /**
     * 返回Layer的偏置
     */
    virtual const std::vector<sftensor>& bias() const;

    /**
     * 返回Layer的名称
     */
    virtual const std::string& layer_name() const { return this->name_; }

    /**
     * 设置Layer对应的计算节点
     * @param op 计算节点
     */
    void set_runtime_op(const srunop& op);

protected:
    std::string name_;  // Layer的名称——与计算节点类型名不一样

    std::weak_ptr<RuntimeOperator> op_;  // Layer对应的计算节点——用weak_ptr避免循环引用
};

using slayer = std::shared_ptr<Layer>;

}  // namespace TinyInfer

#endif  // TINY_INFER_SOURCE_LAYER_LAYER_HPP_
