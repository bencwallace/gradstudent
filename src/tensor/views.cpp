#include <sstream>

#include "tensor.h"

Tensor flatten(const Tensor &tensor) {
  return Tensor(array_t{tensor.size()}, array_t{1}, tensor);
}

Tensor permute(const Tensor &tensor, std::initializer_list<size_t> axes) {
  if (axes.size() != tensor.ndims()) {
    std::stringstream ss;
    ss << "Expected axis list of length " << tensor.ndims() << ", got "
       << axes.size();
    throw std::invalid_argument(ss.str());
  }

  const array_t &shape = tensor.shape();
  const array_t &strides = tensor.strides();

  array_t result_shape(tensor.ndims());
  array_t result_strides(tensor.ndims());
  size_t i = 0;
  for (size_t axis : axes) {
    result_shape[i] = shape[axis];
    result_strides[i] = strides[axis];
    ++i;
  }

  return Tensor(result_shape, result_strides, tensor);
}

Tensor Tensor::slice(const array_t &mIdx) {
  if (mIdx.size() > ndims()) {
    std::stringstream ss;
    ss << "Multi-index of size " << mIdx.size()
       << " too large for tensor of rank " << ndims();
    throw std::invalid_argument(ss.str());
  }

  size_t result_ndims = ndims() - mIdx.size();
  array_t result_shape(result_ndims);
  array_t result_strides(result_ndims);
  for (size_t i = mIdx.size(); i < ndims(); ++i) {
    result_shape[i - mIdx.size()] = shape_[i];
    result_strides[i - mIdx.size()] = strides_[i];
  }

  return Tensor(result_shape, result_strides, *this, toIndex(mIdx));
}
