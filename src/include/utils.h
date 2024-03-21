#pragma once

#include <ostream>
#include <vector>

#include "types.h"

namespace gradstudent {

class Tensor;

enum { BCAST_LEFT = -1, BCAST_NONE = 0, BCAST_RIGHT = 1 };

/* ARITHMETIC */

array_t defaultStrides(const array_t &shape);

template <typename P, typename Q>
size_t sumProd(const P &left, const Q &right, size_t start, size_t end) {
  size_t sum = 0;
  for (size_t i = start; i < end; ++i) {
    sum += left[i] * right[i];
  }
  return sum;
}

template <typename P, typename Q>
size_t sumProd(const P &left, const Q &right) {
  return sumProd(left, right, 0, std::min(left.size(), right.size()));
}

template <typename T> size_t prod(const T &array) {
  size_t result = 1;
  for (size_t i = 0; i < array.size(); ++i) {
    result *= array[i];
  }
  return result;
}

// returned mask entry true if left dimension broadcast
std::vector<int> broadcastShapes(array_t &out, const array_t &left,
                                 const array_t &right);

void checkCompatibleShape(const Tensor &, const Tensor &);

/* STREAMS */

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &array) {
  std::ostream &result = os << "(";
  for (int i = 0; i < (int)array.size() - 1; ++i) {
    result << array[i] << ", ";
  }
  if (!array.empty()) {
    result << array[array.size() - 1];
  }
  if (array.size() == 1) {
    result << ",";
  }
  result << ")";
  return result;
}

} // namespace gradstudent
