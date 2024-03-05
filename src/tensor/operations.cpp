#include <sstream>

#include "multiIndex.h"
#include "tensor.h"

Tensor Tensor::dot(const Tensor &other) const {
  if (shape[shape.size - 1] != other.shape[0]) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << shape << " and " << other.shape;
    throw std::invalid_argument(ss.str());
  }

  Array result_shape(shape.size + other.shape.size - 2);
  for (size_t i = 0; i < shape.size - 1; ++i) {
    result_shape[i] = shape[i];
  }
  for (size_t i = 1; i < other.shape.size; ++i) {
    result_shape[shape.size + i - 2] = other.shape[i];
  }

  Tensor result(result_shape);
  MultiIndex resultMultiIndex = MultiIndex(result.shape);
  for (size_t i = 0; i < result.size; ++i) {
    size_t thisIndex = toIndex(resultMultiIndex, 0, ndims - 1);
    size_t otherIndex = other.toIndex(resultMultiIndex, ndims - 1, result.ndims);

    result.data[i] = 0;
    for (size_t j = 0; j < shape[shape.size - 1]; ++j) {
      result.data[i] += (*this)[thisIndex] * other[otherIndex];;
      thisIndex += strides[ndims - 1];
      otherIndex += other.strides[0];
    }

    ++resultMultiIndex;
  }

  return result;
}

Tensor Tensor::flatten() const {
  return Tensor(size, 1, Array({size}), Array({1}), data);
}

Tensor Tensor::permute(std::initializer_list<size_t> axes) {
  if (axes.size() != ndims) {
    std::stringstream ss;
    ss << "Expected axis list of length " << ndims << ", got " << axes.size();
    throw std::invalid_argument(ss.str());
  }

  Array result_shape(ndims);
  Array result_strides(ndims);
  size_t i = 0;
  for (size_t axis : axes) {
    result_shape[i] = shape[axis];
    result_strides[i] = strides[axis];
    ++i;
  }

  return Tensor(size, ndims, result_shape, result_strides, data);
}
