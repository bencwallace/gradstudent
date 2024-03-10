#include <sstream>

#include "tensor.h"

/* PRIVATE */

Tensor::Tensor(const Array &shape, const Array &strides)
    : size_(shape.prod()), ndims_(shape.size), shape_(shape), strides_(strides),
      data_(new TensorDataCpu(size_)) {}

Tensor::Tensor(const Array &shape, const Array &strides, const Tensor &tensor)
    : size_(shape.prod()), ndims_(shape.size), shape_(shape), strides_(strides), data_(tensor.data_) {}

/* PUBLIC */

Tensor::Tensor(const Array &shape)
    : size_(shape.prod()), ndims_(shape.size), shape_(shape), strides_(shape.size),
      data_(new TensorDataCpu(size_)) {
  if (ndims_ > 0) {
    strides_[ndims_ - 1] = 1;
  }
  for (int i = (int) ndims_ - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * this->shape_[i + 1];
  }
}

Tensor::Tensor(const Array &shape, std::initializer_list<double> data)
    : Tensor(shape) {
  if (data.size() != this->data_->size()) {
    std::stringstream ss;
    ss << "Expected " << this->data_->size() << " data values, got " << data.size();
    throw std::invalid_argument(ss.str());
  }
  size_t i = 0;
  for (double val : data) {
    (*this->data_)[i++] = val;
  }
}

Tensor::Tensor(double value)
    : Tensor({}, {value}) {}
