#include "relu.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <glog/logging.h>
#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.hpp"
#endif

namespace TinyInfer {

ReLU::ReLU() : NoAttrLayer("ReLU") {}

InferStatus ReLU::Forward(const std::vector<sftensor>& inputs, std::vector<sftensor>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array is empty";
    return InferStatus::InferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  const uint32_t batch = inputs.size();

// OpenMP多线程并行处理同一批次内不同输入Tensor
#pragma omp parallel for num_threads(batch)
  for (uint32_t b = 0; b < batch; ++b) {
  // 检查每个输入输出Tensor是否为空
    const sftensor& input = inputs.at(b);
    sftensor& output = outputs.at(b);
    
    CHECK(input != nullptr && !input->empty()) 
        << "The " << b << "th/st/nd input tensor is empty";

    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The " << b << "th/st/nd output tensor is empty";
      output = std::make_shared<ftensor>(input->shape());
    }

    CHECK(input->shape() == output->shape()) 
        << "The " << b << "th/st/nd input and output tensor shape do not match";

// SSE2指令向量化处理
#if __SSE2__
    const float* in_ptr = input->raw_ptr();
    float* out_ptr = const_cast<float*>(output->raw_ptr());
    const uint32_t size = input->size();  // 获取Tensor中元素总数
    const uint32_t packet_size = 4;
    __m128 _zero = _mm_setzero_ps();  // 将向量寄存器a清0——获得4个值为0.0的float32数
    uint32_t i = 0;
    for ( ; i < size - 3; i += packet_size) {
      __m128 _in = _mm_load_ps(in_ptr);  // 加载4个float32操作数到向量寄存器b中
      __m128 _out = _mm_max_ps(_zero, _in);  // 单条指令比较四个操作数
      _mm_store_ps(out_ptr, _out);  // 把四个计算结果写回内存
      in_ptr += packet_size;
      out_ptr += packet_size;
    }

    // 单独处理剩余不足4个的操作数
    while (i < size) {
      float in = input->index(i);
      output->index(i) = in > 0.f ? in : 0.f;
      i += 1;
    }
#else
    output->set_data(input->data());
    output->Transform([] (const float val) { return val > 0.f ? val : 0.f; });
#endif
  }
  return InferStatus::InferSuccess;
}

ParseParamAttrStatus ReLU::GetInstance(const srunop& op, slayer& relu) {
  if (op == nullptr) {
    LOG(ERROR) << "Operator is empty";
    return ParseParamAttrStatus::OpEmpty;
  }

  relu = std::make_shared<ReLU>();
  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

// 调用LayerRegisterWrapper的构造函数，注册ReLU layer的创建函数GetInstance到注册表中
LayerRegisterWrapper ReluGetInstance("nn.ReLU", ReLU::GetInstance);

}  // namespace TinyInfer
