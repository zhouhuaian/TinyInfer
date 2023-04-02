#ifndef TINY_INFER_DATA_BLOB_HPP_
#define TINY_INFER_DATA_BLOB_HPP_

#include <memory>
#include <vector>
#include <armadillo>

namespace TinyInfer {

// TODO
template <typename T>
class Tensor {};

// TODO
template <>
class Tensor<uint8_t> {};

template <>
class Tensor<float> {
public:
    explicit Tensor() = default;

    /**
     * 创建张量
     * @param channels 通道数
     * @param rows 行数
     * @param cols 列数
     */
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

    /**
     * 创建张量
     * @param shapes 维度
     */
    explicit Tensor(const std::vector<uint32_t>& shapes);

    Tensor(const Tensor& tensor);

    Tensor(Tensor&& tensor) noexcept;

    Tensor<float>& operator=(const Tensor& tensor);

    Tensor<float>& operator=(Tensor&& tensor) noexcept;

    /**
     * 返回张量的通道数
     */
    uint32_t channels() const;

    /**
     * 返回张量的行数
     */
    uint32_t rows() const;

    /**
     * 返回张量的列数
     */
    uint32_t cols() const;

    /**
     * 返回张量的元素数量
     */
    uint32_t size() const;

    /**
     * 设置张量中的元素值
     * @param data 数据
     */
    void set_data(const arma::fcube& data);

    /**
     * 判断张量是否为空
     */
    bool empty() const;

    /**
     * 返回张量的维度
     * @return 维度——（通道数，行数，列数）
     */
    std::vector<uint32_t> shape() const;

    /**
     * 返回张量的实际维度
     * @return 实际维度
     */
    const std::vector<uint32_t>& raw_shapes() const;

    /**
     * 访问张量中offset处的元素
     * @param offset 访问位置
     */
    float& index(uint32_t offset);

    /**
     * 访问张量中offset处的元素
     * @param offset 访问位置
     */
    float index(uint32_t offset) const;

    /**
     * 返回张量中的数据
     */
    arma::fcube& data();

    /**
     * 返回张量中的数据
     */
    const arma::fcube& data() const;

    /**
     * 返回张量中第channel通道的数据
     * @param channel 通道编号
     */
    arma::fmat& slice(uint32_t channel);

    /**
     * 返回张量中第channel通道的数据
     * @param channel 通道数
     */
    const arma::fmat& slice(uint32_t channel) const;

    /**
     * 返回特定位置的元素
     * @param channel 通道数
     * @param row 行数
     * @param col 列数
     */
    float& at(uint32_t channel, uint32_t row, uint32_t col);

    /**
     * 返回特定位置的元素
     * @param channel 通道数
     * @param row 行数
     * @param col 列数
     */
    float at(uint32_t channel, uint32_t row, uint32_t col) const;

    /**
     * 扩充张量维度
     * @param pads 扩充参数
     * @param padding_value 扩充值
     */
    void Padding(const std::vector<uint32_t>& pads, float padding_value);

    /**
     * 填充张量元素值
     * @param value 填充值
     */
    void Fill(float value);

    /**
     * 填充张量元素值
     * @param values 填充值
     * @param row_major 是否依据行主序填充
     */
    void Fill(const std::vector<float>& values, bool row_major = true);

    /**
     * 以常量1初始化张量
     */
    void Ones();

    /**
     * 以随机值初始化张量
     */
    void Rand();

    /**
     * 打印张量
     */
    void Show();

    /**
     * 重排张量元素
     * @param shapes 目标张量的实际维度
     * @param row_major 是否依据行主序进行重排
     */
    void Reshape(const std::vector<uint32_t>& shapes, bool row_major = false);

    /**
     * 展平张量
     * @param 是否依据行主序进行展平
     */
    void Flatten(bool row_major = false);

    /**
     * 返回张量内所有元素
     * @param row_major 是否以行主序返回
     */
    std::vector<float> values(bool row_major = true);

    /**
     * 对张量元素进行转换
     * @param filter 转换函数
     */
    void Transform(const std::function<float(float)>& filter);

    /**
     * 对张量进行深拷贝
     */
    std::shared_ptr<Tensor> Clone();

    /**
     * 获取张量数据的原始指针
     */
    const float* raw_ptr() const;

private:
    /**
     * 以行主序重排张量元素
     * @param shapes 目标张量维度
     */
    void ReView(const std::vector<uint32_t>& shapes);
    
    std::vector<uint32_t> raw_shapes_;  // 张量的实际维度
    arma::fcube data_;                  // 张量数据
};

using ftensor = Tensor<float>;
using sftensor = std::shared_ptr<Tensor<float>>;

/**
 * 比较张量是否相同
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 */
bool TensorIsSame(const sftensor& tensor1, const sftensor& tensor2);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 */
sftensor TensorElementAdd(const sftensor& tensor1, const sftensor& tensor2);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 */
void TensorElementAdd(const sftensor& tensor1, const sftensor& tensor2, 
                      const sftensor& output_tensor);

/**
 * 张量element-wise相乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 */
sftensor TensorElementMultiply(const sftensor& tensor1, const sftensor& tensor2);

/**
 * 张量Element-wise相乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 */
void TensorElementMultiply(const sftensor& tensor1, const sftensor& tensor2,
                           const sftensor& output_tensor);

/**
 * 创建张量
 * @param channels 通道数
 * @param rows 行数
 * @param cols 列数
 */
sftensor TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols);

/**
 * 创建张量
 * @param shapes 张量维度
 */
sftensor TensorCreate(const std::vector<uint32_t>& shapes);

/**
 * 以广播方式扩充张量维度
 * @param tensor1 张量1
 * @param tensor2 张量2
 * @return 维度相同的两个张量
 */
std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& tensor1, const sftensor& tensor2);

/**
 * 扩充张量维度
 * @param tensor 原张量
 * @param pads 扩充参数
 * @param padding_value 扩充值
 * @return 扩充维度的新张量
 */
sftensor TensorPadding(const sftensor& tensor, const std::vector<uint32_t>& pads, 
                       float padding_value);

}  // namespace TinyInfer

#endif  // TINY_INFER_DATA_BLOB_HPP_
