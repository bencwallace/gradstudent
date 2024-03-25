#include <sstream>

#include "multi_index.h"
#include "ops.h"
#include "tensor.h"
#include "tensor_iter.h"

namespace gradstudent {

void Tensor::assignSelf(const Tensor &other) {
  auto temp(std::make_unique<double[]>(size_));
  size_t i = 0;
  for (auto &mIdx : MultiIndexIter(shape_)) {
    temp[i++] = other[mIdx];
  }
  i = 0;
  for (auto &mIdx : MultiIndexIter(shape_)) {
    data_[toIndex(mIdx)] = temp[i++];
  }
}

void Tensor::assignOther(const Tensor &other) {
  for (auto vals : TensorIter(*this, other)) {
    std::get<0>(vals) = std::get<1>(vals);
  }
}

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

} // namespace gradstudent
