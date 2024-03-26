#pragma once

#include <tuple>

#include "array.h"
#include "meta.h"
#include "tensor.h"

namespace gradstudent {

/* ITERATOR */

template <bool... Const> class TensorIter {

public:
  using value_type = bool_to_const_t<double, Const...>;
  using reference = add_ref_t<value_type>;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;

  TensorIter(std::conditional_t<Const, const Tensor, Tensor> &...tensors)
      : tensors_(tensors...), shape_(std::get<0>(tensors_).shape()),
        mIdx_(shape_.size(), 0), isEnd_(false) {
    std::apply([](auto &...indices) { ((indices = 0), ...); }, indices_);
  }

  const array_t &index() const { return mIdx_; }

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

  reference operator*() const {
    return derefHelper(std::make_index_sequence<sizeof...(Const)>{});
  }

  bool operator==(const TensorIter &other) const {
    return indices_ == other.indices_ && isEnd_ == other.isEnd_;
  }

  bool operator!=(const TensorIter &other) const { return !((*this) == other); }

  TensorIter begin() { return *this; }

  TensorIter end() {
    auto it = std::apply(
        [](auto &...tensors) { return TensorIter(tensors...); }, tensors_);
    if (!it.mIdx_.empty()) {
      it.mIdx_[0] = it.shape_[0];
      if (it.mIdx_.size() > 1) {
        std::fill(it.mIdx_.begin() + 1, it.mIdx_.end(), 0);
      }
    }
    it.syncIndicesHelper(std::make_index_sequence<sizeof...(Const)>{});
    it.isEnd_ = true;
    return it;
  }

private:
  std::tuple<std::conditional_t<Const, const Tensor &, Tensor &>...> tensors_;
  ntuple_t<sizeof...(Const), size_t> indices_;
  array_t shape_;
  array_t mIdx_;
  bool isEnd_;

  /* TEMPLATE HELPERS */

  template <size_t... Is> auto syncIndicesHelper(std::index_sequence<Is...>) {
    std::apply(
        [this](auto &...indices) {
          ((indices = std::get<Is>(tensors_).toIndex(mIdx_)), ...);
        },
        indices_);
  }

  template <size_t... Is> auto derefHelper(std::index_sequence<Is...>) const {
    return std::forward_as_tuple(
        std::get<Is>(tensors_).data_[std::get<Is>(indices_)]...);
  }

  template <size_t... Is>
  void incrementHelper(std::index_sequence<Is...> is, size_t currDim) {
    ++mIdx_[currDim];
    std::apply(
        [this, currDim](auto &...indices) {
          ((indices += std::get<Is>(tensors_).strides_[currDim]), ...);
        },
        indices_);

    if (mIdx_[currDim] == shape_[currDim]) {
      if (currDim == 0) {
        isEnd_ = true;
      } else {
        mIdx_[currDim] = 0;
        std::apply(
            [this, currDim](auto &...indices) {
              ((indices -=
                std::get<Is>(tensors_).strides_[currDim] * shape_[currDim]),
               ...);
            },
            indices_);
        incrementHelper(is, currDim - 1);
      }
    }
  }

  template <size_t... Is> void incrementHelper(std::index_sequence<Is...> is) {
    incrementHelper(is, shape_.size() - 1);
  }

  /* RECURSION HELPERS */

  void increment() {
    return incrementHelper(std::make_index_sequence<sizeof...(Const)>{});
  }
};

/* DEDUCTION GUIDES */

template <typename... Args>
TensorIter(Args &...) -> TensorIter<std::is_const_v<Args>...>;

} // namespace gradstudent
