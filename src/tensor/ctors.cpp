#include <sstream>

#include "multi_index.h"
#include "tensor.h"
#include "utils.h"

namespace gradstudent {

// tensor copy constructor
Tensor::Tensor(const Tensor &other) : Tensor(other.shape_) {
  for (MultiIndex mIdx : multiIndexRange()) {
    (*this)[mIdx] = other[mIdx];
  }
}

// tensor view constructor
Tensor::Tensor(const array_t &shape, const array_t &strides,
               const Tensor &tensor, size_t offset, bool ro)
    : ro_(ro), offset_(offset), size_(prod(shape)), shape_(shape),
      strides_(strides), data_(tensor.data_) {}

// empty tensor constructor (default strides)
Tensor::Tensor(const array_t &shape)
    : offset_(0), size_(prod(shape)), shape_(shape),
      strides_(defaultStrides(shape)),
      data_(std::shared_ptr<double[]>(new double[size_])) {}

// non-empty tensor constructor
Tensor::Tensor(const array_t &shape, const array_t &strides,
               std::initializer_list<double> data)
    : offset_(0), size_(prod(shape)), shape_(shape), strides_(strides),
      data_(std::shared_ptr<double[]>(new double[size_])) {
  if (data.size() > 0 && data.size() != size_) {
    std::stringstream ss;
    ss << "Data should be empty or of size " << size_ << ", got size "
       << data.size();
    throw std::invalid_argument(ss.str());
  }
  size_t i = 0;
  for (double val : data) {
    (this->data_)[i++] = val;
  }
}

// non-empty tensor constructor (default strides)
Tensor::Tensor(const array_t &shape, std::initializer_list<double> data)
    : Tensor(shape, defaultStrides(shape), data) {}

// scalar tensor constructor
Tensor::Tensor(double value) : Tensor({}, {value}) {}

} // namespace gradstudent
