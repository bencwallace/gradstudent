#include <sstream>

#include "internal/utils.h"
#include "tensor.h"
#include "tensor_iter.h"

namespace gs {

// tensor copy constructor
Tensor::Tensor(const Tensor &other) : Tensor(other.shape_) {
  for (auto vals : TensorIter(*this, other)) {
    std::get<0>(vals) = std::get<1>(vals);
  }
}

// tensor view constructor
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
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
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Tensor::Tensor(const array_t &shape, const array_t &strides,
               const std::vector<double> &data)
    : offset_(0), size_(prod(shape)), shape_(shape), strides_(strides),
      data_(std::shared_ptr<double[]>(new double[size_])) {
  if (!data.empty() && data.size() != size_) {
    std::stringstream ss;
    ss << "Data should be empty or of size " << size_ << ", got size "
       << data.size();
    throw std::invalid_argument(ss.str());
  }
  size_t i = 0;
  for (double val : data) {
    data_[i++] = val;
  }
}

// non-empty tensor constructor (default strides)
Tensor::Tensor(const array_t &shape, const std::vector<double> &data)
    : Tensor(shape, defaultStrides(shape), data) {}

// scalar tensor constructor
Tensor::Tensor(double value) : Tensor({}, {value}) {}

Tensor Tensor::fill(const array_t &shape, const array_t &strides,
                    double value) {
  return Tensor(shape, strides, std::vector<double>(prod(shape), value));
}

Tensor Tensor::fill(const array_t &shape, double value) {
  return Tensor::fill(shape, defaultStrides(shape), value);
}

} // namespace gs
