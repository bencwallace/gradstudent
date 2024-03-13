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
