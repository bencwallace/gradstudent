#pragma once

#include "array.h"
#include "multi_index.h"

Array defaultStrides(const Array &shape);

size_t sumProd(const Array &, const Array &, size_t start, size_t end);
size_t sumProd(const MultiIndex &, const Array &, size_t start, size_t end);

inline size_t sumProd(const Array &left, const Array &right) {
  return sumProd(left, right, 0, left.size);
}

inline size_t sumProd(const MultiIndex &left, const Array &right) {
  return sumProd(left, right, 0, left.size());
}
