#include <sstream>

#include "kernels.h"
#include "multi_index.h"
#include "ops.h"
#include "tensor.h"

// TODO: special care required when both tensors shared same buffer (e.g.
// copying to self in reversed order)
Tensor &Tensor::operator=(const Tensor &other) {
  ensureWritable();

  if (size_ != other.size_ || shape_ != other.shape_) {
    std::stringstream ss;
    ss << "Can't copy tensor of shape " << other.shape_
       << " into tensor of shape " << shape_;
    throw std::invalid_argument(ss.str());
  }

  for (MultiIndex mIdx : multiIndexRange()) {
    data_[toIndex(mIdx)] = other[mIdx];
  }

  return *this;
}

Tensor::operator double() const {
  if (size() != 1) {
    std::stringstream ss;
    ss << "Expected tensor of size 1, got size " << size();
    throw std::invalid_argument(ss.str());
  }
  return (*this)[0];
}
