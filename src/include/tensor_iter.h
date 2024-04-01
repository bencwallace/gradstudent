#pragma once

#include <tuple>

#include <boost/iterator/iterator_facade.hpp>

#include "array.h"
#include "meta.h"
#include "tensor.h"

namespace gradstudent {

/**
 * @brief Tensor iterator
 *
 * Allows iteration through multiple tensors in lockstep. In other words, the
 * value returned by the iterator is a tuple of values from each tensor at the
 * same multi-index, where multi-indices are incremented in lexicographic order.
 *
 * @tparam Const Pack of boolean values indicating whether the corresponding
 * tensor should be treated as constant.
 */
template <bool... Const>
class TensorIter : public boost::iterator_facade<
                       TensorIter<Const...>, bool_to_const_t<double, Const...>,
                       std::random_access_iterator_tag,
                       add_ref_t<bool_to_const_t<double, Const...>>> {

private:
  using base_type =
      boost::iterator_facade<TensorIter<Const...>,
                             bool_to_const_t<double, Const...>,
                             std::random_access_iterator_tag,
                             add_ref_t<bool_to_const_t<double, Const...>>>;

public:
  /**
   * @brief Construct a new TensorIter object
   *
   * A deduction guide is provided so that each parameter in Const is set
   * according to whether or not the corresponding tensor is constant. If any of
   * the non-constant tensors should be treated as constant, the Const
   * parameters must be specified explicitly.
   *
   * @param tensors Pack of tensors to iterate through. Must have the same
   * shape.
   */
  TensorIter(std::conditional_t<Const, const Tensor, Tensor> &...tensors)
      : tensors_(&tensors...), shape_(std::get<0>(tensors_)->shape()),
        mIdx_(shape_.size(), 0), endIdx_(endIndex()), isEnd_(false) {
    syncIndicesHelper(std::make_index_sequence<sizeof...(Const)>{});
  }

  /**
   * @brief Returns the current multi-index
   */
  const array_t &index() const { return mIdx_; }

  bool operator!=(const TensorIter &other) const { return !((*this) == other); }

  TensorIter begin() { return *this; }

  TensorIter end() {
    auto it = std::apply(
        [](auto &...tensors) { return TensorIter(*tensors...); }, tensors_);
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
  std::tuple<std::conditional_t<Const, const Tensor *, Tensor *>...> tensors_;
  ntuple_t<sizeof...(Const), size_t>
      indices_;   // indices into each tensor's buffer
  array_t shape_; // common tensor shape
  array_t mIdx_;  // current multi-index
  array_t endIdx_;
  bool isEnd_; // needed for scalar case, in which there is a unique (empty)
               // multi-index

  /* ITERATOR FACADE REQUIREMENTS */

  friend class boost::iterator_core_access;

  typename base_type::reference dereference() const {
    return derefHelper(std::make_index_sequence<sizeof...(Const)>{});
  }

  bool equal(const TensorIter &other) const {
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

  template <size_t... Is>
  void advance(typename base_type::difference_type n,
               std::index_sequence<Is...> is) {
    // handle scalar case first
    if (shape_.size() == 0) {
      isEnd_ = n > 0;
      return;
    }

    auto nIdx = n > 0 ? toMultiIndex(n) : toMultiIndex(-n);
    mIdx_ = n > 0 ? mIdx_ + nIdx : mIdx_ - nIdx;
    auto nIndices = toBufferIndices(nIdx, is);
    if (n > 0) {
      std::apply(
          [&, this]() {
            ((std::get<Is>(indices_) += std::get<Is>(nIndices)), ...);
          },
          std::tuple<>{});
    } else {
      std::apply(
          [&, this]() {
            ((std::get<Is>(indices_) -= std::get<Is>(nIndices)), ...);
          },
          std::tuple<>{});
    }

    isEnd_ = mIdx_ >= endIdx_;
  }

  typename base_type::difference_type
  distance_to(const TensorIter &other) const {
    // assume compatible iterators
    if (shape_.size() == 0) {
      return other.isEnd_ - isEnd_;
    }
    return other.numSteps() - numSteps();
  }

  /* TEMPLATE HELPERS */

  template <size_t... Is>
  ntuple_t<sizeof...(Const), size_t>
  toBufferIndices(const array_t &idx, std::index_sequence<Is...>) {
    return std::make_tuple(
        std::inner_product(idx.begin(), idx.end(),
                           std::get<Is>(tensors_)->strides_.begin(), 0)...);
  }

  // Computes each buffer index to match the current multi-index by taking the
  // dot product of the shared multi-index with each tensor's strides.
  template <size_t... Is> auto syncIndicesHelper(std::index_sequence<Is...>) {
    std::apply(
        [this](auto &...indices) {
          ((indices = std::get<Is>(tensors_)->toIndex(mIdx_)), ...);
        },
        indices_);
  }

  // Produces a tuple of references to each tensor's entry at its corresponding
  // buffer index
  template <size_t... Is> auto derefHelper(std::index_sequence<Is...>) const {
    return std::forward_as_tuple(
        std::get<Is>(tensors_)->data_[std::get<Is>(indices_)]...);
  }

  template <size_t... Is>
  void decrementHelper(std::index_sequence<Is...> is, size_t currDim) {
    if (mIdx_[currDim] > 0) {
      --mIdx_[currDim];
      std::apply(
          [this, currDim](auto &...indices) {
            ((indices -= std::get<Is>(tensors_)->strides_[currDim]), ...);
          },
          indices_);
    } else {
      mIdx_[currDim] = shape_[currDim] - 1;
      std::apply(
          [this, currDim](auto &...indices) {
            ((indices +=
              std::get<Is>(tensors_)->strides_[currDim] * mIdx_[currDim]),
             ...);
          },
          indices_);
      if (currDim > 0) {
        decrementHelper(is, currDim - 1);
      }
    }
  }

  template <size_t... Is>
  void incrementHelper(std::index_sequence<Is...> is, size_t currDim) {
    // increment the multi-index along the current dimension
    ++mIdx_[currDim];
    // increment the buffer indices by the corresponding strides
    std::apply(
        [this, currDim](auto &...indices) {
          ((indices += std::get<Is>(tensors_)->strides_[currDim]), ...);
        },
        indices_);

    // handle overflow along current dimension
    if (mIdx_[currDim] == shape_[currDim]) {
      if (currDim == 0) { // end iteration
        isEnd_ = true;
      } else {
        // reset multi-index along current dimension
        mIdx_[currDim] = 0;
        // update buffer indices accordingly
        std::apply(
            [this, currDim](auto &...indices) {
              ((indices -=
                std::get<Is>(tensors_)->strides_[currDim] * shape_[currDim]),
               ...);
            },
            indices_);
        // recursively continue incrementing along the next dimension
        incrementHelper(is, currDim - 1);
      }
    }
  }

  template <size_t... Is> void decrementHelper(std::index_sequence<Is...> is) {
    decrementHelper(is, shape_.size() - 1);
  }

  template <size_t... Is> void incrementHelper(std::index_sequence<Is...> is) {
    incrementHelper(is, shape_.size() - 1);
  }

  /* RECURSION HELPERS */

  void decrementHelper() {
    return decrementHelper(std::make_index_sequence<sizeof...(Const)>{});
  }

  // Lexicographically increments the multi-index and updates the buffer indices
  // accordingly using the corresponding tensors' strides.
  void incrementHelper() {
    return incrementHelper(std::make_index_sequence<sizeof...(Const)>{});
  }

  /* OTHER HELPERS */

  void advance(typename base_type::difference_type n) {
    advance(n, std::make_index_sequence<sizeof...(Const)>{});
  }

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
TensorIter(Args &...) -> TensorIter<std::is_const_v<Args>...>;

template <bool... Const> class ITensorIter {

public:
  using value_type = tuple_cat_t<std::tuple<array_t>,
                                 typename TensorIter<Const...>::value_type>;
  using reference = tuple_cat_t<std::tuple<const array_t &>,
                                typename TensorIter<Const...>::reference>;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;

  ITensorIter(std::conditional_t<Const, const Tensor, Tensor> &...tensors)
      : iter_(tensors...) {}

  ITensorIter &operator++() {
    ++iter_;
    return *this;
  }

  ITensorIter operator++(int) {
    ITensorIter tmp(*this);
    operator++();
    return tmp;
  }

  reference operator*() const {
    return std::tuple_cat(std::forward_as_tuple(iter_.index()), *iter_);
  }

  bool operator==(const ITensorIter &other) const {
    return iter_ == other.iter_;
  }

  bool operator!=(const ITensorIter &other) const {
    return iter_ != other.iter_;
  }

  ITensorIter begin() { return *this; }

  ITensorIter end() { return ITensorIter(iter_.end()); }

private:
  TensorIter<Const...> iter_;

  ITensorIter(const TensorIter<Const...> &iter) : iter_(iter) {}
};

template <typename... Args>
ITensorIter(Args &...) -> ITensorIter<std::is_const_v<Args>...>;

} // namespace gradstudent
