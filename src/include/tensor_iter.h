#pragma once

#include <tuple>

#include "array.h"
#include "meta.h"

namespace gradstudent {

/* ITERATOR */

template <typename... Ts> class TensorIter {

public:
  using value_type = const_convert_t<double, Ts...>;
  using reference = add_ref_t<value_type>;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;

  TensorIter(Ts &...tensors)
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
    return derefHelper(std::index_sequence_for<Ts...>{});
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
    it.syncIndicesHelper(std::index_sequence_for<Ts...>{});
    it.isEnd_ = true;
    return it;
  }

private:
  std::tuple<Ts &...> tensors_;
  ntuple_t<sizeof...(Ts), size_t> indices_;
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

  void increment() { return incrementHelper(std::index_sequence_for<Ts...>{}); }
};

} // namespace gradstudent
