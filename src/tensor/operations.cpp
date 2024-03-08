#include <sstream>

#include "multiIndex.h"
#include "tensor.h"

size_t Tensor::size() const {
  return data.size();
}

const Array &Tensor::shape() const {
  return shape_;
}

Tensor Tensor::dot(const Tensor &other) const {
  if (shape_[shape_.size - 1] != other.shape_[0]) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << shape_ << " and " << other.shape_;
    throw std::invalid_argument(ss.str());
  }

  Array result_shape(shape_.size + other.shape_.size - 2);
  for (size_t i = 0; i < shape_.size - 1; ++i) {
    result_shape[i] = shape_[i];
  }
  for (size_t i = 1; i < other.shape_.size; ++i) {
    result_shape[shape_.size + i - 2] = other.shape_[i];
  }

  Tensor result(result_shape);
  MultiIndex resultMultiIndex = MultiIndex(result.shape_);
  for (size_t i = 0; i < result.size(); ++i) {
    size_t thisIndex = toIndex(resultMultiIndex, 0, ndims - 1);
    size_t otherIndex = other.toIndex(resultMultiIndex, ndims - 1, result.ndims);

    result.data[i] = 0;
    for (size_t j = 0; j < shape_[shape_.size - 1]; ++j) {
      result.data[i] += this->data[thisIndex] * other.data[otherIndex];
      thisIndex += strides[ndims - 1];
      otherIndex += other.strides[0];
    }

    ++resultMultiIndex;
  }

  return result;
}

Tensor Tensor::flatten() const {
  return Tensor(Array({size()}), Array({1}), data);
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
    result_shape[i] = shape_[axis];
    result_strides[i] = strides[axis];
    ++i;
  }

  return Tensor(result_shape, result_strides, data);
}
