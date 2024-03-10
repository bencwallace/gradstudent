#include <sstream>

#include "multi_index.h"
#include "kernels.h"
#include "tensor.h"

Tensor &Tensor::operator=(const Tensor &other) {
  if (size_ != other.size_ || shape_ != other.shape_) {
    std::stringstream ss;
    ss << "Can't copy tensor of shape " << other.shape_ << " into tensor of shape " << shape_;
    throw std::invalid_argument(ss.str());
  }

  MultiIndex mIdx(shape_);
  for (size_t i = 0; i < size_; ++i) {
    (*this)[toIndex(mIdx)] = other[toIndex(mIdx)];
    ++mIdx;
  }

  return *this;
}

double Tensor::operator[](size_t i) const { return (*data_)[offset_ + i]; }

double &Tensor::operator[](size_t i) { return (*data_)[offset_ + i]; }

Tensor Tensor::operator[](const Array &mIdx) const {
  if (mIdx.size > ndims()) {
    std::stringstream ss;
    ss << "Multi-index of size " << mIdx.size << " too large for tensor of rank " << ndims();
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

double &Tensor::operator[](const Array &mIdx) {
  return (*this)[toIndex(mIdx)];
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
  if (size() != other.size() || ndims() != other.ndims() || shape_ != other.shape_) {
    return false;
  }

  MultiIndex mIdx(shape_);
  for (size_t i = 0; i < size(); ++i) {
    if (static_cast<double>((*this)[mIdx]) != static_cast<double>(other[mIdx])) {
      return false;
    }
    ++mIdx;
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
