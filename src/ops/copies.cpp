#include "gradstudent/iter.h"
#include "gradstudent/ops.h"
#include "gradstudent/tensor.h"

namespace gs {

// FLATTEN

Tensor flatten(const Tensor &tensor) {
  auto result = Tensor(array_t{tensor.size()});
  size_t i = 0;
  for (const auto &[x] : TensorIter(tensor)) {
    result[i++] = x;
  }
  return result;
}

} // namespace gs
