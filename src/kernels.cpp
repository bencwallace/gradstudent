#include "kernels.h"
#include "multi_index.h"
#include "utils.h"

void addKernel(Tensor &result, const Tensor &left, const Tensor &right) {
  for (auto mIdx : result.multiIndexRange()) {
    result[mIdx] = left[mIdx] + right[mIdx];
  }
}

void multKernel(Tensor &result, const double scalar, const Tensor &tensor) {
  for (auto mIdx : result.multiIndexRange()) {
    result[mIdx] = scalar * tensor[mIdx];
  }
}

void multKernel(Tensor &result, const Tensor &left, const Tensor &right) {
  for (MultiIndex resultIdx : result.multiIndexRange()) {
    result[resultIdx] = left[resultIdx] * right[resultIdx];
  }
}

void negKernel(Tensor &result, const Tensor &tensor) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    result[i] = -tensor[i];
  }
}

void dotKernel(Tensor &result, const Tensor &left, const Tensor &right) {
  const array_t &left_strides = left.strides();
  const array_t &right_strides = right.strides();

  size_t i = 0;
  for (MultiIndex resultMultiIdx : result.multiIndexRange()) {
    // TODO: find a better way to do this
    size_t thisIndex =
        sumProd(resultMultiIdx, left.strides(), 0, left.ndims() - 1);
    size_t otherIndex = sumProd(resultMultiIdx, right.strides(),
                                left.ndims() - 1, result.ndims());

    result[i] = 0;
    for (size_t j = 0; j < left.shape()[left.ndims() - 1]; ++j) {
      result[i] += left[thisIndex] * right[otherIndex];
      thisIndex += left_strides[left.ndims() - 1];
      otherIndex += right_strides[0];
    }

    ++i;
  }
}
