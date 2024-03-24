#include <iostream>

#include "multi_index.h"
#include "ops.h"
#include "tensor_iter.h"

namespace gradstudent {

Tensor operator+(const Tensor &left, const Tensor &right) {
  auto [x, y] = broadcast(left, right);
  Tensor result(x.shape());
  for (auto mIdx : MultiIndexIter(result.shape())) {
    result[mIdx] = x[mIdx] + y[mIdx];
  }
  return result;
}

Tensor operator*(const Tensor &left, const Tensor &right) {
  auto [x, y] = broadcast(left, right);
  Tensor result(x.shape());
  for (auto &resultIdx : MultiIndexIter(result.shape())) {
    result[resultIdx] = x[resultIdx] * y[resultIdx];
  }
  return result;
}

Tensor operator-(const Tensor &tensor) {
  Tensor result(tensor.shape());
  for (auto vals : TensorTuple(result, tensor)) {
    std::get<0>(vals) = -std::get<1>(vals);
  }
  return result;
}

Tensor operator-(const Tensor &left, const Tensor &right) {
  return left + (-right);
}

bool operator==(const Tensor &left, const Tensor &right) {
  checkCompatibleShape(left, right);
  for (auto vals : TensorTuple(left, right)) {
    if (std::get<0>(vals) != std::get<1>(vals)) {
      return false;
    }
  }
  return true;
}

} // namespace gradstudent