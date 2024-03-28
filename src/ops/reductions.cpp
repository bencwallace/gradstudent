#include "ops.h"
#include "tensor.h"
#include "tensor_iter.h"

namespace gradstudent {

double max(const Tensor &tensor) {
  double result = tensor[0];
  for (const auto &[elem] : TensorIter(tensor)) {
    result = std::max(result, elem);
  }
  return result;
}

double sum(const Tensor &tensor) {
  double result = 0;
  for (const auto &[elem] : TensorIter(tensor)) {
    result += elem;
  }
  return result;
}

} // namespace gradstudent
