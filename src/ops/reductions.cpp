#include "gradstudent/iter.h"
#include "gradstudent/ops.h"
#include "gradstudent/tensor.h"

namespace gs {

size_t argmax(const Tensor &tensor) {
  if (tensor.ndims() != 1) {
    throw std::invalid_argument("Tensor must be 1D");
  }
  size_t index = 0;
  double val = tensor[0];
  for (const auto &[idx, elem] : ITensorIter(tensor)) {
    if (elem > val) {
      index = idx[0];
      val = elem;
    }
  }
  return index;
}

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

} // namespace gs
