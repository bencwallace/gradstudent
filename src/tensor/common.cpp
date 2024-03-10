#include <sstream>

#include "tensor.h"

size_t Tensor::toIndex(const Array &mIdx, size_t start, size_t end) const {
  if (start < 0) {
    std::stringstream ss;
    ss << "Multi-index start point must be non-negative, got " << start;
    throw std::invalid_argument(ss.str());
  }
  if (end > mIdx.size) {
    std::stringstream ss;
    ss << "Invalid end point " << end << " for multi-index of size " << mIdx.size;
    throw std::invalid_argument(ss.str());
  }

  size_t idx = 0;
  for (size_t i = start; i < end; ++i) {
    if (mIdx[i] < 0 || mIdx[i] >= shape_[i]) {
      std::stringstream ss;
      ss << "Expected index " << i << " in [0, " << shape_[i] << "), got: " << mIdx[i];
      throw std::invalid_argument(ss.str());
    }
    idx += (mIdx[i] + offset_[i]) * strides_[i];
  }
  return idx;
}

size_t Tensor::toIndex(const Array &mIdx) const {
  return toIndex(mIdx, 0, ndims_);
}

Array Tensor::toMultiIndex(size_t idx) const {
  // Not guaranteed to work for arrays with non-standard strides
  Array result(ndims_);
  for (size_t i = 0; i < ndims_; ++i) {
    result[i] = idx / strides_[i];
    idx -= result[i] * strides_[i];
  }
  return result;
}

void Tensor::checkCompatibleShape(const Tensor &other) const {
  if (ndims_ != other.ndims_) {
    std::ostringstream ss;
    ss << "Incompatible ranks: " << ndims_ << " and " << other.ndims_;
    throw std::invalid_argument(ss.str());
  }
  if (shape_ != other.shape_) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << shape_ << " and " << other.shape_;
  }
}

size_t Tensor::size() const {
  return size_;
}

size_t Tensor::ndims() const {
  return ndims_;
}

const Array &Tensor::shape() const {
  return shape_;
}

const Array &Tensor::strides() const {
  return strides_;
}
