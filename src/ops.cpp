#include "ops.h"

double norm2(const Tensor &tensor) {
    Tensor flat = tensor.flatten();
    return static_cast<double>(flat.dot(flat.permute({1, 0})));
}
