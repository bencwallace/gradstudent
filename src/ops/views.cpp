#include <sstream>

#include "ops.h"
#include "tensor.h"

namespace gradstudent {

// FLATTEN

Tensor flatten(Tensor &tensor) {
  return Tensor(array_t{tensor.size()}, array_t{1}, tensor);
}

const Tensor flatten(const Tensor &tensor) {
  return Tensor(array_t{tensor.size()}, array_t{1}, tensor, 0, true);
}

// PERMUTE

std::tuple<array_t, array_t> permuteCommon(const Tensor &tensor,
                                           std::initializer_list<size_t> axes) {
  if (axes.size() != tensor.ndims()) {
    std::stringstream ss;
    ss << "Expected axis list of length " << tensor.ndims() << ", got "
       << axes.size();
    throw std::invalid_argument(ss.str());
  }

  const array_t &shape = tensor.shape();
  const array_t &strides = tensor.strides();

  array_t result_shape(tensor.ndims());
  array_t result_strides(tensor.ndims());
  size_t i = 0;
  for (size_t axis : axes) {
    result_shape[i] = shape[axis];
    result_strides[i] = strides[axis];
    ++i;
  }

  return {result_shape, result_strides};
}

Tensor permute(Tensor &tensor, std::initializer_list<size_t> axes) {
  auto [result_shape, result_strides] = permuteCommon(tensor, axes);
  return Tensor(result_shape, result_strides, tensor);
}

const Tensor permute(const Tensor &tensor, std::initializer_list<size_t> axes) {
  auto [result_shape, result_strides] = permuteCommon(tensor, axes);
  return Tensor(result_shape, result_strides, tensor, 0, true);
}

// SLICE

std::pair<array_t, array_t> sliceCommon(const Tensor &tensor,
                                        const array_t &mIdx) {
  size_t ndims = tensor.ndims();
  if (mIdx.size() > ndims) {
    std::stringstream ss;
    ss << "Multi-index of size " << mIdx.size()
       << " too large for tensor of rank " << ndims;
    throw std::invalid_argument(ss.str());
  }

  array_t shape = tensor.shape();
  array_t strides = tensor.strides();

  size_t result_ndims = ndims - mIdx.size();
  array_t result_shape(result_ndims);
  array_t result_strides(result_ndims);
  for (size_t i = mIdx.size(); i < ndims; ++i) {
    result_shape[i - mIdx.size()] = shape[i];
    result_strides[i - mIdx.size()] = strides[i];
  }

  return {result_shape, result_strides};
}

Tensor slice(Tensor &tensor, const array_t &mIdx) {
  auto [result_shape, result_strides] = sliceCommon(tensor, mIdx);
  return Tensor(result_shape, result_strides, tensor, tensor.toIndex(mIdx));
}

const Tensor slice(const Tensor &tensor, const array_t &mIdx) {
  auto [result_shape, result_strides] = sliceCommon(tensor, mIdx);
  return Tensor(result_shape, result_strides, tensor, tensor.toIndex(mIdx),
                true);
}

// BROADCAST

void broadcastStrides(array_t &out_left, array_t &out_right,
                      const std::vector<int> &mask, const array_t &left,
                      const array_t &right) {
  out_left = left;
  out_right = right;
  out_right.insert(out_right.begin(), mask.size() - out_right.size(), 0);
  out_left.insert(out_left.begin(), mask.size() - out_left.size(), 0);

  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i] == BCAST_LEFT) {
      out_left[i] = 0;
    } else if (mask[i] == BCAST_RIGHT) {
      out_right[i] = 0;
    }
  }
}

std::tuple<Tensor, Tensor> broadcast(const Tensor &left, const Tensor &right) {
  array_t shape, left_strides, right_strides;
  auto mask = broadcastShapes(shape, left.shape(), right.shape());
  broadcastStrides(left_strides, right_strides, mask, left.strides(),
                   right.strides());

  return {Tensor(shape, left_strides, left),
          Tensor(shape, right_strides, right)};
}

} // namespace gradstudent
