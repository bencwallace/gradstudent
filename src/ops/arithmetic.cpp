#include <iostream>

#include "ops.h"
#include "tensor_iter.h"

namespace gradstudent {

Tensor operator+(const Tensor &left, const Tensor &right) {
  auto [bleft, bright] = broadcast(left, right);
  Tensor result(bleft.shape());
  for (const auto &[res, lt, rt] : TensorIter(result, bleft, bright)) {
    res = lt + rt;
  }
  return result;
}

Tensor operator*(const Tensor &left, const Tensor &right) {
  auto [bleft, bright] = broadcast(left, right);
  Tensor result(bleft.shape());
  for (const auto &[res, lt, rt] : TensorIter(result, bleft, bright)) {
    res = lt * rt;
  }
  return result;
}

Tensor operator-(const Tensor &tensor) {
  Tensor result(tensor.shape());
  for (const auto &[res, val] : TensorIter(result, tensor)) {
    res = -val;
  }
  return result;
}

Tensor operator-(const Tensor &left, const Tensor &right) {
  return left + (-right);
}

bool operator==(const Tensor &left, const Tensor &right) {
  checkCompatibleShape(left, right);
  for (const auto &[lt, rt] : TensorIter(left, right)) {
    if (lt != rt) {
      return false;
    }
  }
  return true;
}

} // namespace gradstudent