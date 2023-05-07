#ifndef TINY_INFER_SOURCE_KERNEL_ATTR_KERNEL_HPP_
#define TINY_INFER_SOURCE_KERNEL_ATTR_KERNEL_HPP_

#include "kernel.hpp"

namespace TinyInfer {

class AttrKernel : public Kernel {
public:
  explicit AttrKernel(const std::string &name);

  /**
   * 初始化权重
   * @param count 权重数目，比如Conv2d层的卷积核数目
   */
  void InitWeights(const uint32_t count, const uint32_t channel,
                   const uint32_t height, const uint32_t width);

  /**
   * 初始化偏置
   * @param count 偏置数目，比如Conv2d层的卷积核数目
   */
  void InitBias(const uint32_t count, const uint32_t channel,
                const uint32_t height, const uint32_t width);

  void set_weights(const std::vector<sftensor> &weights) override;

  void set_weights(const std::vector<float> &weights) override;

  void set_bias(const std::vector<sftensor> &bias) override;

  void set_bias(const std::vector<float> &bias) override;

  const std::vector<sftensor> &weights() const override;

  const std::vector<sftensor> &bias() const override;

protected:
  std::vector<sftensor> weights_; // 权重
  std::vector<sftensor> bias_;    // 偏置
};

} // namespace TinyInfer

#endif // TINY_INFER_SOURCE_KERNEL_ATTR_KERNEL_HPP_
