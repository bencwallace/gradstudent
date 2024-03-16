#include <sstream>

#include "tensor.h"
#include "utils.h"

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
