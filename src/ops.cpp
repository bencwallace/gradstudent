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

void dotOp(Tensor &result, const Tensor &left, const Tensor &right) {
  const Array &left_strides = left.strides();
  const Array &right_strides = right.strides();

  MultiIndex resultMultiIndex = MultiIndex(result.shape());
  for (size_t i = 0; i < result.size(); ++i) {
    size_t thisIndex = left.toIndex(resultMultiIndex, 0, left.ndims() - 1);
    size_t otherIndex = right.toIndex(resultMultiIndex, left.ndims() - 1, result.ndims());

    result[i] = 0;
    for (size_t j = 0; j < left.shape()[left.ndims() - 1]; ++j) {
      result[i] += left[thisIndex] * right[otherIndex];
      thisIndex += left_strides[left.ndims() - 1];
      otherIndex += right_strides[0];
    }

    ++resultMultiIndex;
  }
}

double norm2(const Tensor &tensor) {
  Tensor flat = tensor.flatten();
  return flat.dot(flat)[{}];
}
