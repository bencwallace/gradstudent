#include "multiIndex.h"
#include "ops.h"

// void addOp(Tensor &result, const Tensor &left, const Tensor &right) {
//   MultiIndex resultIdx(result.shape);
//   for (size_t i = 0; i < left.size; ++i) {
//     result[resultIdx] = left[resultIdx] + right[resultIdx];
//     ++resultIdx;
//   }
// }

double norm2(const Tensor &tensor) {
  Tensor flat = tensor.flatten();
  return flat.dot(flat)[{}];
}
