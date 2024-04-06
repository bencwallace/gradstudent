#include "gradstudent/iter.h"
#include "gradstudent/tensor.h"

#include "gradstudent/internal/utils.h"

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

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Tensor::Tensor(const array_t &shape, const array_t &strides)
    : offset_(0), size_(prod(shape)), shape_(shape), strides_(strides),
      data_(std::shared_ptr<double[]>(new double[size_])) {}

// empty tensor constructor (default strides)
Tensor::Tensor(const array_t &shape) : Tensor(shape, defaultStrides(shape)) {}

// scalar tensor constructor
Tensor::Tensor(double value) : Tensor(array_t{}) { data_[0] = value; }

Tensor Tensor::fill(const array_t &shape, const array_t &strides,
                    double value) {
  auto result = Tensor(shape, strides);
  for (size_t i = 0; i < result.size_; ++i) {
    result[i] = value;
  }
  return result;
}

Tensor Tensor::fill(const array_t &shape, double value) {
  return Tensor::fill(shape, defaultStrides(shape), value);
}

Tensor Tensor::range(int start, int stop, int step) {
  auto result = Tensor(array_t{static_cast<size_t>((stop - start) / step)});
  for (const auto &[x] : TensorIter(result)) {
    x = start;
    start += step;
  }
  return result;
}

Tensor Tensor::range(int start, int stop) { return range(start, stop, 1); }

Tensor Tensor::range(int stop) { return range(0, stop); }

} // namespace gs
