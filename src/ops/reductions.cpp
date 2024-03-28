#include "ops.h"
#include "tensor.h"
#include "tensor_iter.h"

namespace gradstudent {

double sum(const Tensor &tensor) {
  double result = 0;
  for (const auto &[elem] : TensorIter(tensor)) {
    result += elem;
  }
  return result;
}

} // namespace gradstudent
