#include "ops.h"

double norm2(const Tensor &tensor) {
  Tensor flat = tensor.flatten();
  return flat.dot(flat)[{}];
}
