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
 * @brief An iterator across all multi-indices of a given shape
 *
 * Traverses the multi-indices in lexicographic order.
 */
class MultiIndexIter {

public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = array_t;
  using pointer = array_t *;
  using reference = array_t &;

  /* CONSTRUCTORS */

  MultiIndexIter(const MultiIndexIter &);
  MultiIndexIter(const array_t &shape, bool end = false);
  ~MultiIndexIter();

  /* OPERATORS */

  MultiIndexIter &operator=(const MultiIndexIter &);

  reference operator*() const;
  inline pointer operator->() { return curr_; }
  MultiIndexIter &operator++();
  MultiIndexIter operator++(int);
  friend bool operator==(const MultiIndexIter &, const MultiIndexIter &);
  friend bool operator!=(const MultiIndexIter &, const MultiIndexIter &);

  /* GETTERS/SETTERS */

  inline const array_t &shape() { return shape_; }

  /* BEGIN/END */

  MultiIndexIter begin();
  MultiIndexIter end();

private:
  pointer curr_;
  const array_t shape_;
  bool isEnd_; // used for the unique scalar multi-index

  void setToEnd();
  void increment(size_t currDim);
  void increment();
};

inline bool operator!=(const MultiIndexIter &a, const MultiIndexIter &b) {
  return !(a == b);
};

} // namespace gradstudent
