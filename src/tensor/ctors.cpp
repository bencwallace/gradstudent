#include <sstream>

#include "multi_index.h"
#include "tensor.h"
#include "utils.h"

// tensor copy constructor
Tensor::Tensor(const Tensor &other) : Tensor(other.shape_) {
  for (MultiIndex mIdx : multiIndexRange()) {
    (*this)[mIdx] = other[mIdx];
  }
}

// tensor view constructor
Tensor::Tensor(const Array &shape, const Array &strides, const Tensor &tensor,
               size_t offset)
    : offset_(offset), size_(shape.prod()), shape_(shape), strides_(strides),
      data_(tensor.data_) {}

// empty tensor constructor (default strides)
Tensor::Tensor(const Array &shape)
    : offset_(0), size_(shape.prod()), shape_(shape),
      strides_(defaultStrides(shape)), data_(new TensorDataCpu(size_)) {}

// non-empty tensor constructor
Tensor::Tensor(const Array &shape, const Array &strides,
               std::initializer_list<double> data)
    : offset_(0), size_(shape.prod()), shape_(shape), strides_(strides),
      data_(new TensorDataCpu(size_)) {
  if (data.size() > 0 && data.size() != this->data_->size()) {
    std::stringstream ss;
    ss << "Data should be empty or of size " << this->data_->size()
       << ", got size " << data.size();
    throw std::invalid_argument(ss.str());
  }
  size_t i = 0;
  for (double val : data) {
    (*this->data_)[i++] = val;
  }
}

// non-empty tensor constructor (default strides)
Tensor::Tensor(const Array &shape, std::initializer_list<double> data)
    : Tensor(shape, defaultStrides(shape), data) {}

// scalar tensor constructor
Tensor::Tensor(double value) : Tensor({}, {value}) {}
