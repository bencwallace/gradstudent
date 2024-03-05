#include <cstring>
#include <iostream>
#include <sstream>

#include "multiIndex.h"
#include "tensor.h"

/* PRIVATE */

size_t Tensor::toIndex(const Array &multiIndex, size_t start, size_t end) const {
  if (multiIndex.size != ndims) {
    std::stringstream ss;
    ss << "Expected multi-index of size " << ndims << ", got size " << multiIndex.size;
    throw std::invalid_argument(ss.str());
  }

  size_t idx = 0;
  for (size_t i = start; i < end; ++i) {
    if (multiIndex[i] < 0 || multiIndex[i] >= shape[i]) {
      std::stringstream ss;
      ss << "Expected index " << i << " in [0, " << shape[i] << "), got: " << multiIndex[i];
      throw std::invalid_argument(ss.str());
    }
    idx += multiIndex[i] * strides[i];
  }
  return idx;
}

size_t Tensor::toIndex(const Array &multiIndex) const {
  return toIndex(multiIndex, 0, ndims);
}

Array Tensor::toMultiIndex(size_t idx) const {
  // Not guaranteed to work for arrays with non-standard strides
  Array result(ndims);
  for (size_t i = 0; i < ndims; ++i) {
    result[i] = idx / strides[i];
    idx -= result[i] * strides[i];
  }
  return result;
}

void Tensor::checkCompatibleShape(const Tensor &other) const {
  if (ndims != other.ndims) {
    std::ostringstream ss;
    ss << "Incompatible ranks: " << ndims << " and " << other.ndims;
    throw std::invalid_argument(ss.str());
  }
  if (shape != other.shape) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << shape << " and " << other.shape;
  }
}

/* PUBLIC */

// CONSTRUCTORS AND DESTRUCTORS

Tensor::Tensor(size_t size, size_t ndims, const Array &shape, const Array &strides)
    : size(size), ndims(ndims), shape(shape), strides(strides),
      data(std::shared_ptr<double[]>(new double[size])) {}

Tensor::Tensor(const Array &shape)
    : ndims(shape.size), shape(shape), strides(ndims) {
  if (ndims > 0) {
    strides[ndims - 1] = 1;
    size = this->shape[ndims - 1];
  }
  for (int i = (int) ndims - 2; i >= 0; --i) {
    size *= this->shape[i];
    strides[i] = strides[i + 1] * this->shape[i + 1];
  }
  data = std::shared_ptr<double[]>(new double[size]);
}

Tensor::Tensor(std::initializer_list<size_t> shape)
    : Tensor(Array(shape)) {}

Tensor::~Tensor() {}

// OPERATORS

double Tensor::operator[](size_t i) const { return data[i]; }

double &Tensor::operator[](size_t i) { return data[i]; }

Tensor Tensor::operator+(const Tensor &other) const {
  checkCompatibleShape(other);
  Tensor result(size, ndims, shape, strides);
  for (size_t i = 0; i < size; ++i) {
    result.data[i] = data[i] + other.data[i];
  }
  return result;
}

Tensor Tensor::operator-() const {
  Tensor result(size, ndims, shape, strides);
  for (size_t i = 0; i < size; ++i) {
    result.data[i] = -data[i];
  }
  return result;
}

Tensor Tensor::operator-(const Tensor &other) const {
  return (*this) + (-other);
}

// OTHER METHODS

Tensor Tensor::dot(const Tensor &other) const {
  if (shape[shape.size - 1] != other.shape[0]) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << shape << " and " << other.shape;
    throw std::invalid_argument(ss.str());
  }

  // TODO: simplify array initialization
  Array result_shape(shape.size + other.shape.size - 2);
  for (size_t i = 0; i < shape.size - 1; ++i) {
    result_shape[i] = shape[i];
  }
  for (size_t i = 1; i < other.shape.size; ++i) {
    result_shape[shape.size + i - 2] = other.shape[i];
  }

  Tensor result(result_shape);
  MultiIndex resultMultiIndex = MultiIndex(result.shape);
  for (size_t i = 0; i < result.size; ++i) {
    size_t thisIndex = toIndex(resultMultiIndex, 0, ndims - 1);
    size_t otherIndex = other.toIndex(resultMultiIndex, 1, ndims);

    result.data[i] = 0;
    for (size_t j = 0; j < shape[shape.size - 1]; ++j) {
      result.data[i] += (*this)[thisIndex] * other[otherIndex];;
      thisIndex += strides[ndims - 1];
      otherIndex += other.strides[0];
    }

    ++resultMultiIndex;
  }

  return result;
}

/* FRIEND FUNCTIONS */

Tensor operator*(double scalar, const Tensor &tensor) {
  Tensor result(tensor.size, tensor.ndims, tensor.shape, tensor.strides);
  for (size_t i = 0; i < result.size; ++i) {
    result.data[i] = scalar * tensor.data[i];
  }
  return result;
}

Tensor scalarTensor(double scalar) {
  Tensor result({1});
  result[0] = scalar;
  return result;
}
