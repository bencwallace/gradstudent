#include "utils.h"

Array defaultStrides(const Array &shape) {
  Array strides(shape.size);
  if (shape.size > 0) {
    strides[shape.size - 1] = 1;
  }
  for (int i = (int)shape.size - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

size_t sumProd(const Array &left, const Array &right, size_t start,
               size_t end) {
  size_t result = 0;
  for (size_t i = start; i < end; ++i) {
    result += left[i] * right[i];
  }
  return result;
}

size_t sumProd(const MultiIndex &left, const Array &right, size_t start,
               size_t end) {
  size_t result = 0;
  for (size_t i = start; i < end; ++i) {
    result += left[i] * right[i];
  }
  return result;
}
