#include <sstream>

#include "multiIndex.h"
#include "ops.h"
#include "tensor.h"

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
  dotOp(result, *this, other);
  return result;
}

Tensor Tensor::flatten() const {
  return Tensor(Array({size()}), Array({1}), data);
}

Tensor Tensor::permute(std::initializer_list<size_t> axes) {
  if (axes.size() != ndims_) {
    std::stringstream ss;
    ss << "Expected axis list of length " << ndims_ << ", got " << axes.size();
    throw std::invalid_argument(ss.str());
  }

  Array result_shape(ndims_);
  Array result_strides(ndims_);
  size_t i = 0;
  for (size_t axis : axes) {
    result_shape[i] = shape_[axis];
    result_strides[i] = strides_[axis];
    ++i;
  }

  return Tensor(result_shape, result_strides, data);
}
