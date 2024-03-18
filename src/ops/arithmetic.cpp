#include "multi_index.h"
#include "ops.h"

namespace gradstudent {

Tensor operator+(const Tensor &left, const Tensor &right) {
  checkCompatibleShape(left, right);
  Tensor result(left.shape());
  for (auto mIdx : result.multiIndexRange()) {
    result[mIdx] = left[mIdx] + right[mIdx];
  }
  return result;
}

Tensor operator*(const Tensor &left, const Tensor &right) {
  checkCompatibleShape(left, right);
  Tensor result(left.shape());
  for (MultiIndex resultIdx : result.multiIndexRange()) {
    result[resultIdx] = left[resultIdx] * right[resultIdx];
  }
  return result;
}

Tensor operator-(const Tensor &tensor) {
  Tensor result(tensor.shape());
  for (size_t i = 0; i < tensor.size(); ++i) {
    result[i] = -tensor[i];
  }
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
  for (auto mIdx : result.multiIndexRange()) {
    result[mIdx] = scalar * tensor[mIdx];
  }
  return result;
}

} // namespace gradstudent