#include <cstring>
#include <iostream>
#include <sstream>

#include "tensor.h"

/* PRIVATE */

Tensor::Tensor(double scalar) : Tensor({1}) { data[0] = scalar; }

size_t Tensor::toIndex(const Array &multiIndex) const {
  if (multiIndex.size != ndims) {
    std::stringstream ss;
    ss << "Expected multi-index of size " << ndims << ", got size " << multiIndex.size;
    throw std::invalid_argument(ss.str());
  }

  size_t idx = 0;
  for (size_t i = 0; i < ndims; ++i) {
    if (multiIndex[i] < 0 || multiIndex[i] >= shape[i]) {
      std::stringstream ss;
      ss << "Expected index " << i << " in [0, " << shape[i] << "), got: " << multiIndex[i];
      throw std::invalid_argument(ss.str());
    }
    idx += multiIndex[i] * strides[i];
  }
  return idx;
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
  if (ndims != other.ndims) {
    std::ostringstream ss;
    ss << "Incompatible ranks: " << ndims << " and " << other.ndims;
    throw std::invalid_argument(ss.str());
  }
  if (shape != other.shape) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << shape << " and " << other.shape;
  }

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
  Array resultMultiIndex(result_shape);
  for (size_t i = 0; i < result.ndims; ++i) {
    resultMultiIndex[i] = 0;
  }
  size_t currResultDim = 0;
  for (size_t i = 0; i < result.size; ++i) {
    Array thisMultiIndex(ndims);
    for (size_t k = 0; k < ndims - 1; ++k) {
      thisMultiIndex[k] = resultMultiIndex[k];
    }
    thisMultiIndex[ndims - 1] = 0;

    Array otherMultiIndex(other.ndims);
    otherMultiIndex[0] = 0;
    for (size_t k = 1; k < other.ndims; ++k)  {
      otherMultiIndex[k] = resultMultiIndex[ndims + k - 2];
    }

    // TODO: get these directly from slices of resultMultiIndex
    size_t thisIndex = toIndex(thisMultiIndex);
    size_t otherIndex = other.toIndex(otherMultiIndex);

    result.data[i] = 0;
    for (size_t j = 0; j < shape[shape.size - 1]; ++j) {
      result.data[i] += (*this)[thisIndex] * other[otherIndex];;
      thisIndex += strides[ndims - 1];
      otherIndex += other.strides[0];
    }

    if (resultMultiIndex[currResultDim] < result_shape[currResultDim] - 1) {
      ++resultMultiIndex[currResultDim];
    } else {
      resultMultiIndex[currResultDim] = 0;
      ++currResultDim;
    }
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
