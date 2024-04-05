#include <numeric>
#include <sstream>

#include "ops.h"
#include "tensor_iter.h"

namespace gs {

Tensor dot(const Tensor &left, const Tensor &right) {
  const array_t &left_shape = left.shape();
  const array_t &right_shape = right.shape();

  if (left_shape[left_shape.size() - 1] != right_shape[0]) {
    std::ostringstream ss;
    ss << "Incompatible shapes: " << left_shape << " and " << right_shape;
    throw std::invalid_argument(ss.str());
  }

  array_t result_shape =
      sliceTo(left_shape, left_shape.size() - 1) | sliceFrom(right_shape, 1);
  Tensor result(result_shape);
  const array_t &left_strides = left.strides();
  const array_t &right_strides = right.strides();

  for (auto res_it = TensorIter(result); res_it != res_it.end(); ++res_it) {
    auto res_idx = res_it.index();
    size_t thisIndex = std::inner_product(res_idx.begin(),
                                          res_idx.begin() + (left.ndims() - 1),
                                          left_strides.begin(), 0);
    size_t otherIndex = std::inner_product(
        res_idx.rbegin(), res_idx.rbegin() + (right.ndims() - 1),
        right_strides.rbegin(), 0);

    auto [res_val] = *res_it;
    res_val = 0;
    for (size_t j = 0; j < left.shape()[left.ndims() - 1]; ++j) {
      res_val += left[thisIndex] * right[otherIndex];
      thisIndex += left_strides[left.ndims() - 1];
      otherIndex += right_strides[0];
    }
  }

  return result;
}

Tensor norm2(const Tensor &tensor) {
  Tensor flat = flatten(tensor);
  return dot(flat, flat);
}

} // namespace gs
