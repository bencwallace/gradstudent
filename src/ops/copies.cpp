#include <sstream>

#include "ops.h"
#include "tensor.h"

namespace gradstudent {

// FLATTEN

Tensor flatten(const Tensor &tensor) {
  auto result = Tensor(array_t{tensor.size()});
  for (size_t i = 0; i < tensor.size(); ++i) {
    result[i] = tensor[i];
  }
  return result;
}

} // namespace gradstudent
