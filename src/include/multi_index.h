#pragma once

#include <memory>

#include "types.h"

class MultiIndex {

private:
  const std::unique_ptr<size_t[]> data_;
  bool isEnd_;
  const array_t shape_;

  void increment(size_t);

public:
  MultiIndex(const MultiIndex &);
  MultiIndex(const array_t &);

  MultiIndex &operator=(const MultiIndex &);

  inline bool isEnd() const { return isEnd_; }
  void setToEnd();

  inline const array_t &shape() const { return shape_; }
  inline size_t size() const { return shape_.size(); };

  inline size_t operator[](size_t i) const { return data_[i]; }
  inline size_t &operator[](size_t i) { return data_[i]; }

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

  MultiIndexIter(const MultiIndexIter &);
  MultiIndexIter(const array_t &shape, bool end = false);
  ~MultiIndexIter();

  MultiIndexIter &operator=(const MultiIndexIter &);

  reference operator*() const;
  inline pointer operator->() { return curr; }
  MultiIndexIter &operator++();
  MultiIndexIter operator++(int);
  friend bool operator==(const MultiIndexIter &, const MultiIndexIter &);
  friend bool operator!=(const MultiIndexIter &, const MultiIndexIter &);

  inline const array_t &shape() { return curr->shape(); }

  MultiIndexIter begin();
  MultiIndexIter end();

private:
  pointer curr;
};

inline bool operator!=(const MultiIndexIter &a, const MultiIndexIter &b) {
  return !(a == b);
};
