#include "ops.h"

namespace gradstudent {

Tensor relu(const Tensor &tensor) {
  Tensor result(tensor.shape());
  for (size_t i = 0; i < tensor.size(); i++) {
    result[i] = std::max(0.0, tensor[i]);
  }

  return result;
}

} // namespace gradstudent
