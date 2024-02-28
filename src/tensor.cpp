#include <cstring>
#include <iostream>
#include <sstream>

#include "tensor.h"

Tensor::Tensor(double scalar) : Tensor({1}) { data[0] = scalar; }

Tensor::Tensor(size_t size, size_t ndims, Array shape, Array strides)
    : size(size), ndims(ndims), shape(shape), strides(strides),
      data(std::shared_ptr<double[]>(new double[size])) {}

Tensor::Tensor(std::initializer_list<size_t> shape)
    : ndims(shape.size()), shape(shape), strides(ndims) {
  if (ndims > 0) {
    strides[0] = 1;
    size = this->shape[0];
  }
  for (size_t i = 1; i < ndims; ++i) {
    size *= this->shape[i];
    strides[i] = strides[i - 1] * this->shape[i - 1];
  }
  data = std::shared_ptr<double[]>(new double[size]);
}

Tensor::~Tensor() {}

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
