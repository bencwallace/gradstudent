#include <sstream>

#include "multi_index.h"
#include "kernels.h"
#include "tensor.h"

double Tensor::operator[](size_t i) const { return (*data_)[i]; }

double &Tensor::operator[](size_t i) { return (*data_)[i]; }

Tensor Tensor::operator[](const Array &mIdx) const {
  return (*this)[toIndex(mIdx)];
}

double &Tensor::operator[](const Array &mIdx) {
  return (*this)[toIndex(mIdx)];
}

Tensor Tensor::operator+(const Tensor &other) const {
  checkCompatibleShape(other);
  Tensor result(shape_, strides_);
  addKernel(result, *this, other);
  return result;
}

Tensor Tensor::operator*(const Tensor &other) const {
  checkCompatibleShape(other);
  Tensor result(shape_, strides_);
  multKernel(result, *this, other);
  return result;
}

Tensor Tensor::operator-() const {
  Tensor result(shape_, strides_);
  negKernel(result, *this);
  return result;
}

Tensor Tensor::operator-(const Tensor &other) const {
  return (*this) + (-other);
}

bool Tensor::operator==(const Tensor &other) const {
  if (size() != other.size() || ndims_ != other.ndims_ || shape_ != other.shape_) {
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
  Tensor result(tensor.shape_, tensor.strides_);
  multKernel(result, scalar, tensor);
  return result;
}
