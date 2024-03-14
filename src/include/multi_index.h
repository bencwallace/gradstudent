#pragma once

#include "array.h"

class MultiIndex {

private:
  std::unique_ptr<size_t[]> data_;
  bool isEnd_;

  void increment(size_t);

public:
  const Array shape;
  const Array strides;
  const size_t offset;

  MultiIndex(const MultiIndex &);
  MultiIndex(const Array &, const Array &, const size_t);

  MultiIndex &operator=(const MultiIndex &);

  inline bool isEnd() const { return isEnd_; }
  void setToEnd();
  inline size_t toIndex() const { return toIndex(0, size()); }
  size_t toIndex(size_t, size_t) const;

  inline size_t size() const { return shape.size; };
  inline const std::unique_ptr<size_t[]> &data() const { return data_; }

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
  MultiIndexIter(const Array &shape, const Array &strides, size_t offset,
                 bool end = false);
  ~MultiIndexIter();

  MultiIndexIter &operator=(const MultiIndexIter &);

  reference operator*() const;
  inline pointer operator->() { return curr; }
  MultiIndexIter &operator++();
  MultiIndexIter operator++(int);
  friend bool operator==(const MultiIndexIter &, const MultiIndexIter &);
  friend bool operator!=(const MultiIndexIter &, const MultiIndexIter &);

  inline const Array &shape() { return curr->shape; }
  inline const Array &strides() { return curr->strides; }
  inline size_t offset() { return curr->offset; }

  MultiIndexIter begin();
  MultiIndexIter end();

private:
  pointer curr;
};

inline bool operator!=(const MultiIndexIter &a, const MultiIndexIter &b) {
  return !(a == b);
};
