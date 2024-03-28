#include <sstream>

#include "ops.h"
#include "tensor_iter.h"

namespace gradstudent {

Tensor singleConv(const Tensor &input, const Tensor &kernel) {
  array_t offset = kernel.shape() / 2;
  array_t result_shape =
      input.shape() - kernel.shape() + array_t(kernel.ndims(), 1);

  Tensor result(result_shape);
  for (auto [resIdx, res] : ITensorIter(result)) {
    Tensor window(truncate(input, resIdx, resIdx + kernel.shape()));
    res = 0;
    for (auto [win, ker] : TensorIter(window, kernel)) {
      res += win * ker;
    }
  }

  return result;
}

Tensor conv(const Tensor &input, const Tensor &kernel) {
  if (kernel.ndims() < input.ndims()) {
    std::stringstream ss;
    ss << "Input rank should not exceed kernel rank, got " << input.ndims()
       << " and " << kernel.ndims();
    throw std::invalid_argument(ss.str());
  }
  if (kernel.ndims() > 1 + input.ndims()) {
    std::stringstream ss;
    ss << "Kernel rank should not exceed input rank by more than 1. Got kernel "
          "rank "
       << kernel.ndims() << " and input rank " << input.ndims();
    throw std::invalid_argument(ss.str());
  }

  if (kernel.ndims() == input.ndims()) {
    return singleConv(input, kernel);
  }

  array_t singleKernelShape(kernel.shape().begin() + 1, kernel.shape().end());
  array_t singleResultShape =
      input.shape() - singleKernelShape + array_t(input.ndims(), 1);
  array_t resultShape(kernel.ndims());
  resultShape[0] = kernel.shape()[0];
  std::copy(singleResultShape.begin(), singleResultShape.end(),
            resultShape.begin() + 1);

  Tensor result(resultShape);
  for (size_t i = 0; i < kernel.shape()[0]; ++i) {
    slice(result, {i}) = singleConv(input, slice(kernel, array_t{i}));
  }
  return result;
}

} // namespace gradstudent
