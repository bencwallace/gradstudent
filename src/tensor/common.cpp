#include <sstream>

#include "tensor.h"
#include "utils.h"

size_t Tensor::toIndex(const Array &mIdx) const {
  // return toIndex(mIdx, 0, mIdx.size);
  return sumProd(mIdx, strides_);
}

MultiIndexRange Tensor::multiIndexRange() const {
  return MultiIndexRange(shape_, strides_, offset_);
}

Array Tensor::toMultiIndex(size_t idx) const {
  // Not guaranteed to work for arrays with non-standard strides
  Array result(ndims());
  for (size_t i = 0; i < ndims(); ++i) {
    result[i] = idx / strides_[i];
    idx -= result[i] * strides_[i];
  }
  return result;
}

void Tensor::checkCompatibleShape(const Tensor &other) const {
  if (ndims() != other.ndims()) {
    std::ostringstream ss;
    ss << "Incompatible ranks: " << ndims() << " and " << other.ndims();
    throw std::invalid_argument(ss.str());
  }
  if (shape_ != other.shape_) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << shape_ << " and " << other.shape_;
  }
}

size_t Tensor::size() const { return size_; }

size_t Tensor::ndims() const { return shape_.size; }

const Array &Tensor::shape() const { return shape_; }

const Array &Tensor::strides() const { return strides_; }
