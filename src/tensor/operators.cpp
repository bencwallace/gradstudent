#include <sstream>

#include "multi_index.h"
#include "kernels.h"
#include "tensor.h"

double Tensor::operator[](size_t i) const { return (*data_)[i]; }

double &Tensor::operator[](size_t i) { return (*data_)[i]; }

double Tensor::operator[](const Array &multiIndex) const {
  return (*this)[toIndex(multiIndex)];
}

double &Tensor::operator[](const Array &multiIndex) {
  return (*this)[toIndex(multiIndex)];
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

/* FRIEND FUNCTIONS */

Tensor operator*(double scalar, const Tensor &tensor) {
  Tensor result(tensor.shape_, tensor.strides_);
  multKernel(result, scalar, tensor);
  return result;
}
