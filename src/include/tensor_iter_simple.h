#pragma once

#include <tuple>

#include <boost/iterator/iterator_facade.hpp>

#include "array.h"
#include "tensor.h"

namespace gradstudent {

template <bool Const>
class TensorIterSimple
    : public boost::iterator_facade<
          TensorIterSimple<Const>,
          std::conditional_t<Const, const double, double>,
          std::random_access_iterator_tag,
          std::conditional_t<Const, const double &, double &>> {

private:
  using base_type = boost::iterator_facade<
      TensorIterSimple<Const>, std::conditional_t<Const, const double, double>,
      std::random_access_iterator_tag,
      std::conditional_t<Const, const double &, double &>>;

public:
  TensorIterSimple(std::conditional_t<Const, const Tensor, Tensor> &tensor)
      : tensor_(&tensor), shape_(tensor_->shape()), mIdx_(shape_.size(), 0),
        endIdx_(endIndex()), isEnd_(false) {
    syncIndicesHelper();
  }

  const array_t &index() const { return mIdx_; }

  TensorIterSimple begin() { return *this; }

  TensorIterSimple end() {
    auto it = TensorIterSimple(*tensor_);
    if (!it.mIdx_.empty()) {
      it.mIdx_[0] = it.shape_[0];
      if (it.mIdx_.size() > 1) {
        std::fill(it.mIdx_.begin() + 1, it.mIdx_.end(), 0);
      }
    }
    it.syncIndicesHelper();
    it.isEnd_ = true;
    return it;
  }

private:
  std::conditional_t<Const, const Tensor *, Tensor *> tensor_;
  size_t idx_;
  array_t shape_;
  array_t mIdx_;
  array_t endIdx_;
  bool isEnd_;

  /* ITERATOR FACADE REQUIREMENTS */

  friend class boost::iterator_core_access;

  typename base_type::reference dereference() const { return derefHelper(); }

  bool equal(const TensorIterSimple &other) const {
    return mIdx_ == other.mIdx_ && isEnd_ == other.isEnd_;
  }

  void increment() {
    if (!shape_.empty()) {
      incrementHelper();
    } else {
      isEnd_ = true;
    }
  }

  void decrement() {
    isEnd_ = false;
    if (shape_.size() > 0) {
      decrementHelper();
    }
  }

  void advance(typename base_type::difference_type n) {
    advance(n, std::make_index_sequence<sizeof...(Const)>{});
  }

  typename base_type::difference_type
  distance_to(const TensorIterSimple &other) const {
    // assume compatible iterators
    if (shape_.size() == 0) {
      return other.isEnd_ - isEnd_;
    }
    return other.numSteps() - numSteps();
  }

  /* TEMPLATE HELPERS */

  void advance(typename base_type::difference_type n) {
    // handle scalar case first
    if (shape_.size() == 0) {
      isEnd_ = n > 0;
      return;
    }

    auto nIdx = n > 0 ? toMultiIndex(n) : toMultiIndex(-n);
    mIdx_ = n > 0 ? mIdx_ + nIdx : mIdx_ - nIdx;
    auto nIndices = toBufferIndices(nIdx, is);
    if (n > 0) {
      idx_ += nIndices;
    } else {
      idx_ -= nIndices;
    }

    isEnd_ = mIdx_ >= endIdx_;
  }

  size_t toBufferIndices(const array_t &idx) {
    return std::inner_product(idx.begin(), idx.end(), tensor_->strides_.begin(),
                              0);
  }

  void syncIndicesHelper() { idx_ = tensor->toIndex(mIdx_); }

  std::conditional<Const, const double &, double &> derefHelper() const {
    return tensor_->data_[idx_];
  }

  void decrementHelper(size_t currDim) {
    if (mIdx_[currDim] > 0) {
      --mIdx_[currDim];
      idx_ -= tensor_->strides_[currDim];
    } else {
      mIdx_[currDim] = shape_[currDim] - 1;
      idx_ += tensor_->strides_[currDim] * mIdx_[currDim];
      if (currDim > 0) {
        decrementHelper(currDim - 1);
      }
    }
  }

  template <size_t... Is> void incrementHelper(size_t currDim) {
    ++mIdx_[currDim];
    idx_ += tensor_->strides_[currDim];

    if (mIdx_[currDim] == shape_[currDim]) {
      if (currDim == 0) {
        isEnd_ = true;
      } else {
        mIdx_[currDim] = 0;
        idx_ -= tensor_->strides_[currDim] * shape_[currDim];
        incrementHelper(currDim - 1);
      }
    }
  }

  /* RECURSION HELPERS */

  void decrementHelper() { return decrementHelper(shape_.size() - 1); }

  void incrementHelper() { return incrementHelper(shape_.size() - 1); }

  /* OTHER HELPERS */

  array_t endIndex() const {
    if (shape_.size() == 0) {
      return shape_;
    }
    array_t result(shape_.size());
    result[0] = shape_[0];
    return result;
  }

  size_t numSteps() const {
    size_t n = 0;
    size_t runningProd = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
      n += mIdx_[i] * runningProd;
      runningProd *= shape_[i];
    }
    return n;
  }

  array_t toMultiIndex(size_t n) const {
    array_t result(mIdx_.size(), 0);
    result[shape_.size() - 1] = n % shape_[shape_.size() - 1];
    n -= result[shape_.size() - 1];
    for (int i = shape_.size() - 2; i >= 0; --i) {
      result[i] = n / shape_[i + 1];
    }
    return result;
  }
};

template <typename... Args>
TensorIterSimple(Args &...) -> TensorIterSimple<std::is_const_v<Args>...>;
