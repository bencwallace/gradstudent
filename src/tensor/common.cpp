#include <sstream>

#include "tensor.h"
#include "utils.h"

MultiIndexIter Tensor::multiIndexRange() const {
  return MultiIndexIter(shape_);
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

size_t Tensor::ndims() const { return shape_.size(); }

const array_t &Tensor::shape() const { return shape_; }

const array_t &Tensor::strides() const { return strides_; }
