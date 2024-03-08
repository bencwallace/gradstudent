#include "multi_index.h"
#include "kernels.h"

void addKernel(Tensor &result, const Tensor &left, const Tensor &right) {
  MultiIndex resultIdx(result.shape());
  for (size_t i = 0; i < left.size(); ++i) {
    result[resultIdx] = left[resultIdx] + right[resultIdx];
    ++resultIdx;
  }
}

void multKernel(Tensor &result, const double scalar, const Tensor &tensor) {
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = scalar * tensor[i];
  }
}

void multKernel(Tensor &result, const Tensor &left, const Tensor &right) {
  MultiIndex resultIdx(result.shape());
  for (size_t i = 0; i < left.size(); ++i) {
    result[resultIdx] = left[resultIdx] * right[resultIdx];
    ++resultIdx;
  }
}

void negKernel(Tensor &result, const Tensor &tensor) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    result[i] = -tensor[i];
  }
}

void dotKernel(Tensor &result, const Tensor &left, const Tensor &right) {
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