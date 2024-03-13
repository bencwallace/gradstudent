#include <sstream>

#include "tensor.h"

Tensor Tensor::slice(const Array &mIdx) const {
  if (mIdx.size > ndims()) {
    std::stringstream ss;
    ss << "Multi-index of size " << mIdx.size
       << " too large for tensor of rank " << ndims();
    throw std::invalid_argument(ss.str());
  }

  size_t result_ndims = ndims() - mIdx.size;
  Array result_shape(result_ndims);
  Array result_strides(result_ndims);
  for (size_t i = mIdx.size; i < ndims(); ++i) {
    result_shape[i - mIdx.size] = shape_[i];
    result_strides[i - mIdx.size] = strides_[i];
  }

  return Tensor(result_shape, result_strides, *this, toIndex(mIdx));
}
