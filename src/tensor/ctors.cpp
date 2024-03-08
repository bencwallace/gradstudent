#include <sstream>

#include "tensor.h"

/* PRIVATE */

Tensor::Tensor(const Array &shape, const Array &strides)
    : ndims_(shape.size), shape_(shape), strides_(strides), data_(shape.prod()) {}

Tensor::Tensor(const Array &shape, const Array &strides, const TensorData &data)
    : ndims_(shape.size), shape_(shape), strides_(strides), data_(data) {}

/* PUBLIC */

Tensor::Tensor(const Array &shape)
    : ndims_(shape.size), shape_(shape), strides_(shape.size), data_(shape.prod()) {
  if (ndims_ > 0) {
    strides_[ndims_ - 1] = 1;
  }
  for (int i = (int) ndims_ - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * this->shape_[i + 1];
  }
}

Tensor::Tensor(const Array &shape, std::initializer_list<double> data)
    : Tensor(shape) {
  if (data.size() != this->data_.size()) {
    std::stringstream ss;
    ss << "Expected " << this->data_.size() << " data values, got " << data.size();
    throw std::invalid_argument(ss.str());
  }
  size_t i = 0;
  for (double val : data) {
    this->data_[i++] = val;
  }
}

Tensor::Tensor(std::initializer_list<size_t> shape)
    : Tensor(Array(shape)) {}

Tensor::Tensor(std::initializer_list<size_t> shape, std::initializer_list<double> data)
    : Tensor(Array(shape), data) {}

Tensor::Tensor(double value)
    : Tensor({}, {value}) {}
