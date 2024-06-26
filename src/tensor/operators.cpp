#include <sstream>

#include "gradstudent/iter.h"
#include "gradstudent/tensor.h"

namespace gs {

void Tensor::assignSelf(const Tensor &other) {
  auto temp(std::make_unique<double[]>(size_));
  size_t i = 0;
  for (const auto &[val] : TensorIter(other)) {
    temp[i++] = val;
  }
  i = 0;
  for (const auto &[val] : TensorIter(*this)) {
    val = temp[i++];
  }
}

void Tensor::assignOther(const Tensor &other) {
  for (const auto &[res, val] : TensorIter(*this, other)) {
    res = val;
  }
}

// NOLINTNEXTLINE(bugprone-unhandled-self-assignment)
Tensor &Tensor::operator=(const Tensor &other) {
  if (size_ != other.size_ || shape_ != other.shape_) {
    std::stringstream ss;
    ss << "Can't copy tensor of shape " << other.shape_
       << " into tensor of shape " << shape_;
    throw std::invalid_argument(ss.str());
  }

  ensureWritable();
  if (data_ != other.data_) {
    assignOther(other);
  } else {
    assignSelf(other);
  }

  return *this;
}

Tensor::operator double() const {
  if (size() != 1) {
    std::stringstream ss;
    ss << "Expected tensor of size 1, got size " << size();
    throw std::invalid_argument(ss.str());
  }
  return (*this)[0];
}

} // namespace gs
