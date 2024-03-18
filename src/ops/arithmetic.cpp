#include "kernels.h"
#include "multi_index.h"
#include "ops.h"

namespace gradstudent {

Tensor operator+(const Tensor &left, const Tensor &right) {
  checkCompatibleShape(left, right);
  Tensor result(left.shape());
  addKernel(result, left, right);
  return result;
}

Tensor operator*(const Tensor &left, const Tensor &right) {
  checkCompatibleShape(left, right);
  Tensor result(left.shape());
  multKernel(result, left, right);
  return result;
}

Tensor operator-(const Tensor &tensor) {
  Tensor result(tensor.shape());
  negKernel(result, tensor);
  return result;
}

Tensor operator-(const Tensor &left, const Tensor &right) {
  return left + (-right);
}

bool operator==(const Tensor &left, const Tensor &right) {
  checkCompatibleShape(left, right);
  for (MultiIndex mIdx : MultiIndexIter(left.shape())) {
    if (left[mIdx] != right[mIdx]) {
      return false;
    }
  }

  return true;
}

Tensor operator*(double scalar, const Tensor &tensor) {
  Tensor result(tensor.shape_);
  multKernel(result, scalar, tensor);
  return result;
}

} // namespace gradstudent