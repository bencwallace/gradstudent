#include <sstream>

#include "kernels.h"
#include "ops.h"

Tensor dot(const Tensor &left, const Tensor &right) {
  const Array &left_shape = left.shape();
  const Array &right_shape = right.shape();

  if (left_shape[left_shape.size - 1] != right_shape[0]) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << left_shape << " and " << right_shape;
    throw std::invalid_argument(ss.str());
  }

  Array result_shape(left_shape.size + right_shape.size - 2);
  for (size_t i = 0; i < left_shape.size - 1; ++i) {
    result_shape[i] = left_shape[i];
  }
  for (size_t i = 1; i < right_shape.size; ++i) {
    result_shape[left_shape.size + i - 2] = right_shape[i];
  }

  Tensor result(result_shape);
  dotKernel(result, left, right);
  return result;
}

Tensor flatten(const Tensor &tensor) {
  return Tensor(Array({tensor.size()}), Array({1}), tensor);
}

Tensor permute(const Tensor &tensor, std::initializer_list<size_t> axes) {
  if (axes.size() != tensor.ndims()) {
    std::stringstream ss;
    ss << "Expected axis list of length " << tensor.ndims() << ", got " << axes.size();
    throw std::invalid_argument(ss.str());
  }

  const Array &shape = tensor.shape();
  const Array &strides = tensor.strides();

  Array result_shape(tensor.ndims());
  Array result_strides(tensor.ndims());
  size_t i = 0;
  for (size_t axis : axes) {
    result_shape[i] = shape[axis];
    result_strides[i] = strides[axis];
    ++i;
  }

  return Tensor(result_shape, result_strides, tensor);
}

double norm2(const Tensor &tensor) {
  Tensor flat = flatten(tensor);
  return static_cast<double>(dot(flat, flat));
}
