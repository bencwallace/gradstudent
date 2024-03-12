#pragma once

#include "array.h"

class MultiIndex : public Array {

private:
  const Array shape;

  void increment(size_t);

public:
  MultiIndex(const Array &);

  void reset();
  bool operator==(const MultiIndex &) const;
  bool operator!=(const MultiIndex &) const;
  MultiIndex operator++();
};


class MultiIndexRange {

private:
  Array shape;

public:
  struct MultiIndexIter {
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = MultiIndex;
    using pointer           = MultiIndex*;
    using reference         = MultiIndex&;

    MultiIndexIter(const Array &shape, bool end = false);
    ~MultiIndexIter();

    reference operator*() const;
    pointer operator->();
    MultiIndexIter& operator++();
    MultiIndexIter operator++(int);
    friend bool operator==(const MultiIndexIter &, const MultiIndexIter &);
    friend bool operator!=(const MultiIndexIter &, const MultiIndexIter &);

  private:
    pointer curr;
  };

  MultiIndexRange(const Array &shape);

  MultiIndexIter begin();
  MultiIndexIter end();

};
