#include <sstream>

#include "ops.h"
#include "tensor_iter.h"

namespace gradstudent {

Tensor conv(const Tensor &input, const Tensor &kernel) {
  const array_t &input_shape = input.shape();
  const array_t &kernel_shape = kernel.shape();

  if (input.ndims() != kernel.ndims()) {
    std::stringstream ss;
    ss << "Input and kernel should have the same rank, got " << input.ndims()
       << " and " << kernel.ndims();
    throw std::invalid_argument(ss.str());
  }

  array_t offset = kernel_shape / 2;
  array_t result_shape =
      input_shape - kernel_shape + array_t(kernel_shape.size(), 1);

  Tensor result(result_shape);
  for (auto [resIdx, res] : ITensorIter(result)) {
    res = 0;
    for (auto [kerIdx, ker] : ITensorIter(kernel)) {
      res += input[resIdx + kerIdx] * ker;
    }
  }

  return result;
}

} // namespace gradstudent
