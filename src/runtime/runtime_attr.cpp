#include "runtime/runtime_attr.hpp"

namespace TinyInfer {

void RuntimeAttribute::clear() {
  if (!this->weight_data.empty()) {
    std::vector<char> tmp = std::vector<char>();
    this->weight_data.swap(tmp);
  }
}

}  // namespace TinyInfer