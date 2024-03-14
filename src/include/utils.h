#pragma once

#include <cstddef>
#include <memory>
#include <ostream>
#include <vector>

#include "multi_index.h"
#include "types.h"

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

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &array) {
  std::ostream &result = os << "(";
  for (int i = 0; i < (int)array.size() - 1; ++i) {
    result << array[i] << ", ";
  }
  if (array.size() > 0) {
    result << array[array.size() - 1];
  }
  if (array.size() == 1) {
    result << ",";
  }
  result << ")";
  return result;
}
