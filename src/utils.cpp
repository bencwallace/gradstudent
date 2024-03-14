#include "utils.h"

array_t defaultStrides(const array_t &shape) {
  array_t strides(shape.size());
  if (shape.size() > 0) {
    strides[shape.size() - 1] = 1;
  }
  for (int i = (int)shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

size_t sumProd(const array_t &ptr, const array_t &array, size_t start,
               size_t end) {
  size_t sum = 0;
  for (size_t i = start; i < end; ++i) {
    sum += ptr[i] * array[i];
  }
  return sum;
}

size_t sumProd(const array_t &left, const array_t &right) {
  return sumProd(left, right, 0, std::min(left.size(), right.size()));
}

size_t sumProd(const std::unique_ptr<const size_t[]> &ptr, const array_t &array,
               size_t start, size_t end) {
  size_t sum = 0;
  for (size_t i = start; i < end; ++i) {
    sum += ptr[i] * array[i];
  }
  return sum;
}

size_t sumProd(const std::unique_ptr<const size_t[]> &ptr,
               const array_t &array) {
  return sumProd(ptr, array, 0, array.size());
}

size_t sumProd(const std::unique_ptr<size_t[]> &ptr, const array_t &array,
               size_t start, size_t end) {
  size_t sum = 0;
  for (size_t i = start; i < end; ++i) {
    sum += ptr[i] * array[i];
  }
  return sum;
}

size_t sumProd(const std::unique_ptr<size_t[]> &ptr, const array_t &array) {
  return sumProd(ptr, array, 0, array.size());
}

size_t prod(const array_t &array) {
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

template std::ostream &operator<<(std::ostream &, const std::vector<size_t> &);
