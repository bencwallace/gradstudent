#include <sstream>

#include "multiIndex.h"
#include "ops.h"
#include "tensor.h"

/* PRIVATE */

double Tensor::operator[](size_t i) const { return data[i]; }

double &Tensor::operator[](size_t i) { return data[i]; }

/* PUBLIC */

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
  Tensor result(shape, strides);
  MultiIndex resultIdx(result.shape);
  for (size_t i = 0; i < size(); ++i) {
    result[resultIdx] = (*this)[resultIdx] + other[resultIdx];
    ++resultIdx;
  }
  return result;
}

Tensor Tensor::operator*(const Tensor &other) const {
  checkCompatibleShape(other);
  Tensor result(shape, strides);
  MultiIndex resultIdx(result.shape);
  for (size_t i = 0; i < size(); ++i) {
    result[resultIdx] = (*this)[toIndex(resultIdx)] * other[other.toIndex(resultIdx)];
    ++resultIdx;
  }
  return result;
}

Tensor Tensor::operator-() const {
  Tensor result(shape, strides);
  for (size_t i = 0; i < size(); ++i) {
    result.data[i] = -data[i];
  }
  return result;
}

Tensor Tensor::operator-(const Tensor &other) const {
  return (*this) + (-other);
}

/* FRIEND FUNCTIONS */

Tensor operator*(double scalar, const Tensor &tensor) {
  Tensor result(tensor.shape, tensor.strides);
  for (size_t i = 0; i < result.size(); ++i) {
    result.data[i] = scalar * tensor.data[i];
  }
  return result;
}
