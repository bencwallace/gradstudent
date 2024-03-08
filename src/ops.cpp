#include "multiIndex.h"
#include "ops.h"

void addOp(Tensor &result, const Tensor &left, const Tensor &right) {
  MultiIndex resultIdx(result.shape());
  for (size_t i = 0; i < left.size(); ++i) {
    result[resultIdx] = left[resultIdx] + right[resultIdx];
    ++resultIdx;
  }
}

void multOp(Tensor &result, const double scalar, const Tensor &tensor) {
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = scalar * tensor[i];
  }
}

void multOp(Tensor &result, const Tensor &left, const Tensor &right) {
  MultiIndex resultIdx(result.shape());
  for (size_t i = 0; i < left.size(); ++i) {
    result[resultIdx] = left[resultIdx] * right[resultIdx];
    ++resultIdx;
  }
}

void negOp(Tensor &result, const Tensor &tensor) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    result[i] = -tensor[i];
  }
}

double norm2(const Tensor &tensor) {
  Tensor flat = tensor.flatten();
  return flat.dot(flat)[{}];
}
