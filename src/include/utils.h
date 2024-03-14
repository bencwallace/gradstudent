#pragma once

#include "array.h"
#include "multi_index.h"

Array defaultStrides(const Array &shape);

size_t sumProd(const Array &, const Array &, size_t start, size_t end);

inline size_t sumProd(const Array &left, const Array &right) {
  return sumProd(left, right, 0, left.size);
}

inline size_t sumProd(const MultiIndex &left, const MultiIndex &right,
                      size_t start, size_t end) {
  return sumProd(left.data(), right.data(), start, end);
}

inline size_t sumProd(const MultiIndex &left, const MultiIndex &right) {
  return sumProd(left.data(), right.data());
}
