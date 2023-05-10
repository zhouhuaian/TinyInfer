#ifndef TINY_INFER_SOURCE_KERNEL_RELU_HPP_
#define TINY_INFER_SOURCE_KERNEL_RELU_HPP_

#include "kernel/abstract/no_attr_kernel.hpp"

namespace TinyInfer {

class ReLU : public NoAttrKernel {
public:
  explicit ReLU();

  /**
   * 计算函数——实现ReLU kernel的计算
   * @param inputs 输入Tensor
   * @param outputs 输出Tensor
   */
  InferStatus Forward(const std::vector<sftensor> &inputs,
                      std::vector<sftensor> &outputs) override;

  /**
   * 解析op，获得kernel的参数和权重，创建ReLU kernel
   * @param op 计算图节点
   * @param relu 创建的ReLU kernel
   */
  static ParseParamAttrStatus Creator(const srunop &op, skernel &relu);
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_RELU_HPP_
