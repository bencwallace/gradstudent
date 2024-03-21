#pragma once

#include <memory>

#include "types.h"

namespace gradstudent {

class MultiIndex {

private:
  const std::unique_ptr<size_t[]> // NOLINT(cppcoreguidelines-avoid-c-arrays)
      data_;
  bool isEnd_;
  const array_t shape_;

  void increment(size_t);

public:
  /* CONSTRUCTORS */

  MultiIndex(const MultiIndex &);
  MultiIndex(const array_t &);

  ~MultiIndex() = default;

  /* GETTERS/SETTERS */

  void setToEnd();

  inline bool isEnd() const { return isEnd_; }
  inline const array_t &shape() const { return shape_; }
  inline size_t size() const { return shape_.size(); };

  /* OPERATORS */

  inline size_t operator[](size_t i) const { return data_[i]; }
  inline size_t &operator[](size_t i) { return data_[i]; }

  operator array_t() const;

  MultiIndex &operator=(const MultiIndex &);
  bool operator==(const MultiIndex &) const;
  inline bool operator!=(const MultiIndex &other) const {
    return !((*this) == other);
  }

  MultiIndex operator++();
};

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
