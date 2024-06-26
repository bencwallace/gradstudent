/**
 * @file iter.h
 * @author Ben Wallace (me@bcwallace.com)
 * @brief Tensor iterators
 * @version 0.1
 * @date 2024-04-22
 * 
 * @copyright Copyright (c) 2024
 * 
 * @todo It would be better to have a single TensorTuple container, which would hold
 * a tuple of references to Tensor objects. Then the various tensor iterators could
 * be obtained by calling appropriate begin/end methods on this object. Moreover,
 * the tensor iterator objects should hold raw pointers instead of references so they
 * can be default-constructed.
 */
#pragma once

#include <tuple>

#include "gradstudent/array.h"
#include "gradstudent/internal/meta.h"

#include "gradstudent/tensor.h"

namespace gs {

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
template <bool... Const> class TensorIter {

public:
  /** @brief Tuple of possibly const tensor element values */
  using value_type = bool_to_const_t<double, Const...>;

  /** @brief Tuple of references to possibly const tensor element values */
  using reference = add_ref_t<value_type>;

  // @cond
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;
  // @endcond

  /**
   * @brief Construct a new TensorIter object
   *
   * @param tensors Pack of tensors to iterate through. Must have the same
   * shape.
   */
  TensorIter(std::conditional_t<Const, const Tensor, Tensor> &...tensors)
      : tensors_(tensors...), shape_(std::get<0>(tensors_).shape()),
        mIdx_(shape_.size(), 0) {
    syncIndicesHelper(std::make_index_sequence<sizeof...(Const)>{});
  }

  /** @brief Returns the current multi-index */
  const array_t &index() const { return mIdx_; }

  /** @brief Advances to the next tensor elements */
  TensorIter &operator++() {
    if (!shape_.empty()) {
      increment();
    } else {
      isEnd_ = true;
    }
    return *this;
  }

  /** @overload */
  TensorIter operator++(int) {
    TensorIter tmp(*this);
    operator++();
    return tmp;
  }

  /** @brief Returns the current tensor elements */
  reference operator*() const {
    return derefHelper(std::make_index_sequence<sizeof...(Const)>{});
  }

  /** @brief Returns true if both iterators are at the same multi-index */
  bool operator==(const TensorIter &other) const {
    return mIdx_ == other.mIdx_ && isEnd_ == other.isEnd_;
  }

  // @cond
  bool operator!=(const TensorIter &other) const { return !((*this) == other); }
  // @endcond

  /** @brief Returns a TensorIter at the all-zero multi-index */
  TensorIter begin() { return *this; }

  /** @brief Returns a TensorIter one past the final multi-index for the current shape */
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
  ntuple_t<sizeof...(Const), size_t>
      indices_;   // indices into each tensor's buffer
  array_t shape_; // common tensor shape
  array_t mIdx_;  // current multi-index
  bool isEnd_{};  // needed for scalar case, in which there is a unique (empty)
                  // multi-index

  /* TEMPLATE HELPERS */

  // Computes each buffer index to match the current multi-index by taking the
  // dot product of the shared multi-index with each tensor's strides.
  template <size_t... Is> auto syncIndicesHelper(std::index_sequence<Is...>) {
    std::apply(
        [this](auto &...indices) {
          ((indices = std::get<Is>(tensors_).toIndex(mIdx_)), ...);
        },
        indices_);
  }

  // Produces a tuple of references to each tensor's entry at its corresponding
  // buffer index
  template <size_t... Is> auto derefHelper(std::index_sequence<Is...>) const {
    return std::forward_as_tuple(
        std::get<Is>(tensors_).data_[std::get<Is>(indices_)]...);
  }

  template <size_t... Is>
  void incrementHelper(std::index_sequence<Is...> is, size_t currDim) {
    // increment the multi-index along the current dimension
    ++mIdx_[currDim];
    // increment the buffer indices by the corresponding strides
    std::apply(
        [this, currDim](auto &...indices) {
          ((indices += std::get<Is>(tensors_).strides_[currDim]), ...);
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
                std::get<Is>(tensors_).strides_[currDim] * shape_[currDim]),
               ...);
            },
            indices_);
        // recursively continue incrementing along the next dimension
        incrementHelper(is, currDim - 1);
      }
    }
  }

  template <size_t... Is> void incrementHelper(std::index_sequence<Is...> is) {
    incrementHelper(is, shape_.size() - 1);
  }

  /* RECURSION HELPERS */

  // Lexicographically increments the multi-index and updates the buffer indices
  // accordingly using the corresponding tensors' strides.
  void increment() {
    return incrementHelper(std::make_index_sequence<sizeof...(Const)>{});
  }
};

/**
 * @brief Deduction guide for TensorIter constructor
 * 
 * Sets each parameter in Const according to whether or not the corresponding tensor
 * is constant. If any of the non-constant tensors should be treated as constant, the
 * Const parameters must be specified explicitly.
 */
template <typename... Args>
TensorIter(Args &...) -> TensorIter<std::is_const_v<Args>...>;

/**
 * @brief Indexed tensor iterator
 *
 * Similar to TensorIter but includes the current multi-index as the first
 * element of its tuple of returned values.
 */
template <bool... Const> class ITensorIter {

public:
  /** @brief A tuple type consisting of a multi-index followed by tensor elements */
  using value_type = tuple_cat_t<std::tuple<array_t>,
                                 typename TensorIter<Const...>::value_type>;
  
  /** @brief A tuple type consisting of references to the types of value_type */
  using reference = tuple_cat_t<std::tuple<const array_t &>,
                                typename TensorIter<Const...>::reference>;

  // @cond
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
  // @endcond

private:
  TensorIter<Const...> iter_;

  ITensorIter(const TensorIter<Const...> &iter) : iter_(iter) {}
};

// @cond
template <typename... Args>
ITensorIter(Args &...) -> ITensorIter<std::is_const_v<Args>...>;
// @endcond

} // namespace gs
