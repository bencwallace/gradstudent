#include <sstream>

#include "tensor.h"
#include "utils.h"

void Tensor::checkCompatibleShape(const Tensor &other) const {
  if (ndims() != other.ndims()) {
    std::ostringstream ss;
    ss << "Incompatible ranks: " << ndims() << " and " << other.ndims();
    throw std::invalid_argument(ss.str());
  }
  if (shape_ != other.shape_) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << shape_ << " and " << other.shape_;
  }
}
