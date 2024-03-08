#include <sstream>

#include "tensor.h"

/* PRIVATE */

size_t Tensor::toIndex(const Array &multiIndex, size_t start, size_t end) const {
  if (start < 0) {
    std::stringstream ss;
    ss << "Multi-index start point must be non-negative, got " << start;
    throw std::invalid_argument(ss.str());
  }
  if (end > multiIndex.size) {
    std::stringstream ss;
    ss << "Invalid end point " << end << " for multi-index of size " << multiIndex.size;
    throw std::invalid_argument(ss.str());
  }

  size_t idx = 0;
  for (size_t i = start; i < end; ++i) {
    if (multiIndex[i] < 0 || multiIndex[i] >= shape_[i]) {
      std::stringstream ss;
      ss << "Expected index " << i << " in [0, " << shape_[i] << "), got: " << multiIndex[i];
      throw std::invalid_argument(ss.str());
    }
    idx += multiIndex[i] * strides[i];
  }
  return idx;
}

size_t Tensor::toIndex(const Array &multiIndex) const {
  return toIndex(multiIndex, 0, ndims);
}

Array Tensor::toMultiIndex(size_t idx) const {
  // Not guaranteed to work for arrays with non-standard strides
  Array result(ndims);
  for (size_t i = 0; i < ndims; ++i) {
    result[i] = idx / strides[i];
    idx -= result[i] * strides[i];
  }
  return result;
}

void Tensor::checkCompatibleShape(const Tensor &other) const {
  if (ndims != other.ndims) {
    std::ostringstream ss;
    ss << "Incompatible ranks: " << ndims << " and " << other.ndims;
    throw std::invalid_argument(ss.str());
  }
  if (shape_ != other.shape_) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << shape_ << " and " << other.shape_;
  }
}

/* PRIVATE */

size_t Tensor::size() const {
  return data.size();
}

const Array &Tensor::shape() const {
  return shape_;
}
