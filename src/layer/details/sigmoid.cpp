#include "sigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <glog/logging.h>
#if __SSE2__
#include "sse_mathfun.hpp"
#include <emmintrin.h>
#endif

namespace TinyInfer {

Sigmoid::Sigmoid() : NoAttrLayer("Sigmoid") {}

InferStatus Sigmoid::Forward(const std::vector<sftensor> &inputs,
                             std::vector<sftensor> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array is empty";
    return InferStatus::InferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array batch do not match";
    return InferStatus::InferFailedBatchMatchError;
  }

  const uint32_t batch = inputs.size();

#pragma omp parallel for num_threads(batch)
  for (uint32_t b = 0; b < batch; ++b) {
    const sftensor &input = inputs.at(b);
    sftensor &output = outputs.at(b);

    CHECK(input != nullptr && !input->empty())
        << "The " << b << " input tensor is empty";

    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The " << b << " output tensor is empty";
      output = std::make_shared<ftensor>(input->shape());
    }

    CHECK(input->shape() == output->shape())
        << "The " << b << " input and output tensor shape do not match";

#if __SSE2__
    const float *in_ptr = input->raw_ptr();
    float *out_ptr = const_cast<float *>(output->raw_ptr());
    __m128 _zero = _mm_setzero_ps();
    __m128 _one = _mm_set1_ps(1.f);      // 置4个1.f
    const uint32_t size = input->size(); // 输入Tensor中元素总数
    const uint32_t packet_size = 4;
    uint32_t i = 0;
    for (; i < size - 3; i += packet_size) {
      __m128 _in = _mm_load_ps(in_ptr);
      __m128 _out =
          _mm_div_ps(_one, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _in))));
      _mm_store_ps(out_ptr, _out);
      in_ptr += packet_size;
      out_ptr += packet_size;
    }

    while (i < size) {
      float in = input->index(i);
      output->index(i) = 1.f / (1.f + expf(-in));
      i += 1;
    }
#else
    output->set_data(input->data());
    output->Transform([](const float val) { return 1.f / (1.f + expf(-val)); });
#endif
  }

  return InferStatus::InferSuccess;
}

ParseParamAttrStatus Sigmoid::GetInstance(const srunop &op, slayer &sigmoid) {
  if (op == nullptr) {
    LOG(ERROR) << "Operator is empty";
    return ParseParamAttrStatus::OpEmpty;
  }

  sigmoid = std::make_shared<Sigmoid>();
  return ParseParamAttrStatus::ParamAttrParseSuccess;
}

LayerRegisterWrapper SigmoidGetInstance("nn.Sigmoid", Sigmoid::GetInstance);

} // namespace TinyInfer