#include <sstream>

#include "tensor.h"

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
