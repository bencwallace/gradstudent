#include <sstream>

#include "tensor.h"

/* PRIVATE */

Tensor::Tensor(size_t size, size_t ndims, const Array &shape, const Array &strides)
    : size(size), ndims(ndims), shape(shape), strides(strides),
      data(std::shared_ptr<double[]>(new double[size])) {}

Tensor::Tensor(size_t size, size_t ndims, const Array &shape, const Array &strides, const std::shared_ptr<double[]> data)
    : size(size), ndims(ndims), shape(shape), strides(strides),
      data(data) {}

/* PUBLIC */

Tensor::Tensor(const Array &shape)
    : ndims(shape.size), shape(shape), strides(ndims) {
  if (ndims > 0) {
    strides[ndims - 1] = 1;
    size = this->shape[ndims - 1];
  } else {
    size = 1;
  }
  for (int i = (int) ndims - 2; i >= 0; --i) {
    size *= this->shape[i];
    strides[i] = strides[i + 1] * this->shape[i + 1];
  }
  data = std::shared_ptr<double[]>(new double[size]);
}

Tensor::Tensor(const Array &shape, std::initializer_list<double> data)
    : Tensor(shape) {
  if (data.size() != size) {
    std::stringstream ss;
    ss << "Expected " << size << " data values, got " << data.size();
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
