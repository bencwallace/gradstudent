#pragma once

#include "array.h"

class MultiIndex {

private:
  Array data_;
  const Array shape;
  const Array strides;
  const size_t offset;

  void increment(size_t);

public:
  MultiIndex(const MultiIndex &);
  MultiIndex(const Array &, const Array &, const size_t);

  MultiIndex &operator=(const MultiIndex &);
  void setToEnd();
  size_t toIndex() const;
  size_t toIndex(size_t, size_t) const;

  size_t size() const { return data_.size; };
  Array data() const { return data_; }
  size_t operator[](size_t i) const { return data_[i]; }
  size_t &operator[](size_t i) { return data_[i]; }
  bool operator==(const MultiIndex &) const;
  bool operator!=(const MultiIndex &) const;
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
  pointer operator->();
  MultiIndexIter &operator++();
  MultiIndexIter operator++(int);
  friend bool operator==(const MultiIndexIter &, const MultiIndexIter &);
  friend bool operator!=(const MultiIndexIter &, const MultiIndexIter &);

  MultiIndexIter begin();
  MultiIndexIter end();

private:
  Array shape;
  Array strides;
  size_t offset;

  pointer curr;
};
