#include <algorithm>
#include <sstream>

#include "internal/utils.h"
#include "tensor.h"

namespace gs {

array_t defaultStrides(const array_t &shape) {
  array_t strides(shape.size(), 0);
  if (!shape.empty()) {
    strides[shape.size() - 1] = 1;
  }
  for (int i = (int)shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

std::vector<int> broadcastShapes(array_t &out, const array_t &left,
                                 const array_t &right) {
  out = left.size() > right.size() ? left : right;
  std::vector<int> mask;
  mask.resize(out.size());
  std::fill(mask.begin(), mask.end(),
            left.size() > right.size() ? BCAST_RIGHT : BCAST_LEFT);

  for (size_t i = 0; i < std::min(left.size(), right.size()); ++i) {
    size_t out_idx = out.size() - i - 1;
    size_t left_dim = left[left.size() - i - 1];
    size_t right_dim = right[right.size() - i - 1];

    if (left_dim == right_dim) {
      out[out_idx] = left_dim;
      mask[out_idx] = BCAST_NONE;
    } else if (left_dim == 1) {
      out[out_idx] = right_dim;
      mask[out_idx] = BCAST_LEFT;
    } else if (right_dim == 1) {
      out[out_idx] = left_dim;
      mask[out_idx] = BCAST_RIGHT;
    } else {
      std::stringstream ss;
      ss << "Can't broadcast shapes " << left << " and " << right;
      throw std::invalid_argument(ss.str());
    }
  }

  return mask;
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

} // namespace gs
