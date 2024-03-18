#include <sstream>

#include "tensor.h"
#include "utils.h"

namespace gradstudent {

array_t defaultStrides(const array_t &shape) {
  array_t strides(shape.size());
  if (shape.size() > 0) {
    strides[shape.size() - 1] = 1;
  }
  for (int i = (int)shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

void checkCompatibleShape(const Tensor &left, const Tensor &right) {
  if (left.ndims() != right.ndims()) {
    std::ostringstream ss;
    ss << "Incompatible ranks: " << left.ndims() << " and " << right.ndims();
    throw std::invalid_argument(ss.str());
  }
  if (left.shape() != right.shape()) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << left.shape() << " and " << right.shape();
  }
}

} // namespace gradstudent
