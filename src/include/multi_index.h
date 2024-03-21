/**
 * @file multi_index.h
 * @author Ben Wallace (me@bcwallace.com)
 * @brief Tools for tensor indexing.
 * @version 0.1
 * @date 2024-03-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <cstddef>
#include <iterator>
#include <memory>

#include "types.h"

namespace gradstudent {

/**
 * @brief A set of coordinates bounded by a shape
 *
 */
class MultiIndex {
private:
  const std::unique_ptr<size_t[]> // NOLINT(cppcoreguidelines-avoid-c-arrays)
      data_;
  bool isEnd_; // used for the unique scalar multi-index
  const array_t shape_;

  void increment(size_t);

public:
  /* CONSTRUCTORS */

  // @cond
  MultiIndex(const MultiIndex &);
  MultiIndex(const array_t &);

  ~MultiIndex() = default;

  /* GETTERS/SETTERS */

  void setToEnd();
  inline bool isEnd() const { return isEnd_; }
  // @endcond

  /**
   * @brief Returns the shape bounding the multi-index
   */
  inline const array_t &shape() const { return shape_; }

  /**
   * @brief Returns the number of dimensions of the multi-index
   */
  inline size_t size() const { return shape_.size(); };

  /* OPERATORS */

  /**
   * @brief Subscript operator
   */
  inline size_t operator[](size_t i) const { return data_[i]; }

  /**
   * @overload
   *
   * Allows for assignment to the multi-index
   */
  inline size_t &operator[](size_t i) { return data_[i]; }

  // @cond
  operator array_t() const;
  // @endcond

  /**
   * @brief Assignment operator
   *
   * Only multi-indices of the same shape can be assigned to each other.
   *
   * @todo Should allow different shapes, so long as the ranks are the same and
   * the values are compatible.
   */
  MultiIndex &operator=(const MultiIndex &);

  /**
   * @brief Equality operator
   *
   * Returns true if the multi-indices have the same sizes and values.
   */
  bool operator==(const MultiIndex &) const;
  inline bool operator!=(const MultiIndex &other) const {
    return !((*this) == other);
  }

  /**
   * @brief Prefix increment operator.
   *
   * Increments the multi-index to the next lexicographically ordered
   * multi-index.
   *
   * @todo Implement postfix overload.
   */
  MultiIndex operator++();
};

/**
 * @brief An iterator across all multi-indices of a given shape
 *
 * Traverses the multi-indices in lexicographic order.
 */
class MultiIndexIter {

public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = MultiIndex;
  using pointer = MultiIndex *;
  using reference = MultiIndex &;

  /* CONSTRUCTORS */

  MultiIndexIter(const MultiIndexIter &);
  MultiIndexIter(const array_t &shape, bool end = false);
  ~MultiIndexIter();

  /* OPERATORS */

  MultiIndexIter &operator=(const MultiIndexIter &);

  reference operator*() const;
  inline pointer operator->() { return curr; }
  MultiIndexIter &operator++();
  MultiIndexIter operator++(int);
  friend bool operator==(const MultiIndexIter &, const MultiIndexIter &);
  friend bool operator!=(const MultiIndexIter &, const MultiIndexIter &);

  /* GETTERS/SETTERS */

  inline const array_t &shape() { return curr->shape(); }

  /* BEGIN/END */

  MultiIndexIter begin();
  MultiIndexIter end();

private:
  pointer curr;
};

inline bool operator!=(const MultiIndexIter &a, const MultiIndexIter &b) {
  return !(a == b);
};

} // namespace gradstudent
