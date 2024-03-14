#include <sstream>

#include "kernels.h"
#include "multi_index.h"
#include "tensor.h"

Tensor &Tensor::operator=(const Tensor &other) {
  if (size_ != other.size_ || shape_ != other.shape_) {
    std::stringstream ss;
    ss << "Can't copy tensor of shape " << other.shape_
       << " into tensor of shape " << shape_;
    throw std::invalid_argument(ss.str());
  }

  for (MultiIndex mIdx : multiIndexRange()) {
    (*this)[mIdx] = other[mIdx];
  }

  return *this;
}

Tensor Tensor::operator+(const Tensor &other) const {
  checkCompatibleShape(other);
  Tensor result(shape_);
  addKernel(result, *this, other);
  return result;
}

Tensor Tensor::operator*(const Tensor &other) const {
  checkCompatibleShape(other);
  Tensor result(shape_);
  multKernel(result, *this, other);
  return result;
}

Tensor Tensor::operator-() const {
  Tensor result(shape_);
  negKernel(result, *this);
  return result;
}

Tensor Tensor::operator-(const Tensor &other) const {
  return (*this) + (-other);
}

bool Tensor::operator==(const Tensor &other) const {
  if (size() != other.size() || ndims() != other.ndims() ||
      shape_ != other.shape_) {
    return false;
  }

  for (MultiIndex mIdx : multiIndexRange()) {
    if ((*this)[mIdx] != other[mIdx]) {
      return false;
    }
  }

  return true;
}

Tensor::operator double() const {
  if (size() != 1) {
    std::stringstream ss;
    ss << "Expected tensor of size 1, got size " << size();
    throw std::invalid_argument(ss.str());
  }
  return (*this)[0];
}

/* FRIEND FUNCTIONS */

Tensor operator*(double scalar, const Tensor &tensor) {
  Tensor result(tensor.shape_);
  multKernel(result, scalar, tensor);
  return result;
}
