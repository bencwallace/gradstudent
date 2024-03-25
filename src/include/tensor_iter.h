#pragma once

#include <tuple>

#include "tensor.h"

namespace gradstudent {

template <typename T1, typename T2> class TensorIter {

public:
  using E1 = std::conditional_t<std::is_const_v<T1>, const double, double>;
  using E2 = std::conditional_t<std::is_const_v<T2>, const double, double>;

  using difference_type = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;

  TensorIter(T1 &tensor1, T2 &tensor2, bool end = false)
      : tensors_(std::tie(tensor1, tensor2)), shape_(tensor1.shape()),
        mIdx_(tensor1.ndims(), 0), isEnd_(end) {
    if (shape_ != tensor2.shape()) {
      throw std::invalid_argument("Expected tensors of equal shape");
    }
    if (end) {
      if (!mIdx_.empty()) {
        mIdx_[0] = tensor1.shape()[0];
        if (mIdx_.size() > 1) {
          std::fill(mIdx_.begin() + 1, mIdx_.end(), 0);
        }
      }
    }
    std::get<0>(indices_) = tensor1.toIndex(mIdx_);
    std::get<1>(indices_) = tensor2.toIndex(mIdx_);
  }

  auto operator*() const {
    return std::tie(
        std::forward<E1 &>(std::get<0>(tensors_).data_[std::get<0>(indices_)]),
        std::forward<E2 &>(std::get<1>(tensors_).data_[std::get<1>(indices_)]));
  }

  TensorIter &operator++() {
    if (!shape_.empty()) {
      increment();
    } else {
      isEnd_ = true;
    }
    return *this;
  }

  TensorIter operator++(int) {
    TensorIter tmp(*this);
    operator++();
    return tmp;
  }

  bool operator==(const TensorIter &other) const {
    return isEnd_ == other.isEnd_ && indices_ == other.indices_;
  }

  bool operator!=(const TensorIter &other) const { return !(*this == other); }

  TensorIter begin() { return *this; }

  TensorIter end() {
    return TensorIter(std::get<0>(tensors_), std::get<1>(tensors_), true);
  };

private:
  std::tuple<T1 &, T2 &> tensors_;
  std::tuple<size_t, size_t> indices_;
  array_t shape_;
  array_t mIdx_;
  bool isEnd_;

  void increment(size_t currDim) {
    ++mIdx_[currDim];
    std::get<0>(indices_) += std::get<0>(tensors_).strides_[currDim];
    std::get<1>(indices_) += std::get<1>(tensors_).strides_[currDim];

    if (mIdx_[currDim] == shape_[currDim]) {
      if (currDim == 0) {
        isEnd_ = true;
      } else {
        mIdx_[currDim] = 0;
        std::get<0>(indices_) -=
            std::get<0>(tensors_).strides_[currDim] * shape_[currDim];
        std::get<1>(indices_) -=
            std::get<1>(tensors_).strides_[currDim] * shape_[currDim];
        increment(currDim - 1);
      }
    }
  }

  void increment() { increment(shape_.size() - 1); }
};

} // namespace gradstudent
