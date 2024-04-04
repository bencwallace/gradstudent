#include <sstream>

#include "ops.h"
#include "tensor.h"
#include "tensor_iter.h"

namespace gradstudent {

// FLATTEN

Tensor flatten(const Tensor &tensor) {
  auto result = Tensor(array_t{tensor.size()});
  size_t i = 0;
  for (const auto &[x] : TensorIter(tensor)) {
    result[i++] = x;
  }
  return result;
}

} // namespace gradstudent
