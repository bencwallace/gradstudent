#include <sstream>

#include "kernels.h"
#include "ops.h"

Tensor dot(const Tensor &left, const Tensor &right) {
  const array_t &left_shape = left.shape();
  const array_t &right_shape = right.shape();

  if (left_shape[left_shape.size() - 1] != right_shape[0]) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << left_shape << " and " << right_shape;
    throw std::invalid_argument(ss.str());
  }

  array_t result_shape(left_shape.size() + right_shape.size() - 2);
  for (size_t i = 0; i < left_shape.size() - 1; ++i) {
    result_shape[i] = left_shape[i];
  }
  for (size_t i = 1; i < right_shape.size(); ++i) {
    result_shape[left_shape.size() + i - 2] = right_shape[i];
  }

  Tensor result(result_shape);
  dotKernel(result, left, right);
  return result;
}

Tensor norm2(const Tensor &tensor) {
  Tensor flat = flatten(tensor);
  return dot(flat, flat);
}
