#include <sstream>

#include "tensor.h"

/* PRIVATE */

Tensor::Tensor(const Array &shape, const Array &strides)
    : ndims(shape.size), shape(shape), strides(strides), data(shape.prod()) {}

Tensor::Tensor(const Array &shape, const Array &strides, const TensorData &data)
    : ndims(shape.size), shape(shape), strides(strides), data(data) {}

/* PUBLIC */

Tensor::Tensor(const Array &shape)
    : ndims(shape.size), shape(shape), strides(ndims), data(shape.prod()) {
  if (ndims > 0) {
    strides[ndims - 1] = 1;
  }
  for (int i = (int) ndims - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * this->shape[i + 1];
  }
}

Tensor::Tensor(const Array &shape, std::initializer_list<double> data)
    : Tensor(shape) {
  if (data.size() != this->data.size()) {
    std::stringstream ss;
    ss << "Expected " << this->data.size() << " data values, got " << data.size();
    throw std::invalid_argument(ss.str());
  }
  size_t i = 0;
  for (double val : data) {
    this->data[i++] = val;
  }
}

Tensor::Tensor(std::initializer_list<size_t> shape)
    : Tensor(Array(shape)) {}

Tensor::Tensor(std::initializer_list<size_t> shape, std::initializer_list<double> data)
    : Tensor(Array(shape), data) {}

Tensor::Tensor(double value)
    : Tensor({}, {value}) {}
