#include <sstream>

#include "multi_index.h"
#include "tensor.h"
#include "utils.h"

// tensor copy constructor
Tensor::Tensor(const Tensor &other)
    : Tensor(other.shape_) {
  MultiIndex mIdx(shape_);
  for (size_t i = 0; i < size_; ++i) {
    (*this)[toIndex(mIdx)] = other[other.toIndex(mIdx)];
    ++mIdx;
  }
}

// tensor view constructor
Tensor::Tensor(const Array &shape, const Array &strides, const Tensor &tensor, size_t offset)
    : offset_(offset), size_(shape.prod()), shape_(shape), strides_(strides), data_(tensor.data_) {}

// empty tensor constructor
Tensor::Tensor(const Array &shape, size_t offset)
    : offset_(offset), size_(shape.prod()), shape_(shape), strides_(defaultStrides(shape_)),
      data_(new TensorDataCpu(size_)) {}

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
