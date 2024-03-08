#include <sstream>

#include "multi_index.h"
#include "ops.h"
#include "tensor.h"

double Tensor::operator[](size_t i) const { return data[i]; }

double &Tensor::operator[](size_t i) { return data[i]; }

double Tensor::operator[](const Array &multiIndex) const {
  return (*this)[toIndex(multiIndex)];
}

double &Tensor::operator[](const Array &multiIndex) {
  return (*this)[toIndex(multiIndex)];
}

double Tensor::operator[](std::initializer_list<size_t> coords) const {
  return (*this)[Array(coords)];
}

double &Tensor::operator[](std::initializer_list<size_t> coords) {
  return (*this)[Array(coords)];
}

Tensor Tensor::operator+(const Tensor &other) const {
  checkCompatibleShape(other);
  Tensor result(shape_, strides_);
  addOp(result, *this, other);
  return result;
}

Tensor Tensor::operator*(const Tensor &other) const {
  checkCompatibleShape(other);
  Tensor result(shape_, strides_);
  multOp(result, *this, other);
  return result;
}

Tensor Tensor::operator-() const {
  Tensor result(shape_, strides_);
  negOp(result, *this);
  return result;
}

Tensor Tensor::operator-(const Tensor &other) const {
  return (*this) + (-other);
}

/* FRIEND FUNCTIONS */

Tensor operator*(double scalar, const Tensor &tensor) {
  Tensor result(tensor.shape_, tensor.strides_);
  multOp(result, scalar, tensor);
  return result;
}
