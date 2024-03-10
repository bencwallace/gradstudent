#include <sstream>

#include "tensor.h"

// tensor view constructor
Tensor::Tensor(const Array &shape, const Array &strides, const Tensor &tensor, size_t offset)
    : offset_(offset), size_(shape.prod()), shape_(shape), strides_(strides), data_(tensor.data_) {}

// empty tensor constructor
Tensor::Tensor(const Array &shape)
    : size_(shape.prod()), shape_(shape), strides_(shape.size),
      data_(new TensorDataCpu(size_)) {
  if (ndims() > 0) {
    strides_[ndims() - 1] = 1;
  }
  for (int i = (int) ndims() - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * this->shape_[i + 1];
  }
}

// non-empty tensor constructor
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

// scalar tensor constructor
Tensor::Tensor(double value)
    : Tensor({}, {value}) {}
