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

void broadcastStrides(array_t &out, const std::vector<int> &mask,
                      const array_t &in, int side) {
  out = in;
  out.insert(out.begin(), mask.size() - out.size(), 0);
  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i] == side) {
      out[i] = 0;
    }
  }
}

void broadcastStrides(array_t &out_left, array_t &out_right,
                      const std::vector<int> &mask, const array_t &left,
                      const array_t &right) {
  broadcastStrides(out_left, mask, left, BCAST_LEFT);
  broadcastStrides(out_right, mask, right, BCAST_RIGHT);
}

template <typename T> T broadcast(T &tensor, const array_t &shape) {
  array_t out_shape, out_strides;
  auto mask = broadcastShapes(out_shape, tensor.shape(), shape);
  broadcastStrides(out_strides, mask, tensor.strides(), BCAST_LEFT);
  return Tensor(out_shape, out_strides, tensor, 0, std::is_const<T>::value);
}
template Tensor broadcast<Tensor>(Tensor &, const array_t &);
template const Tensor broadcast<const Tensor>(const Tensor &, const array_t &);

template <typename S, typename T>
std::tuple<S, T> broadcast(S &left, T &right) {
  array_t shape, left_strides, right_strides;
  auto mask = broadcastShapes(shape, left.shape(), right.shape());
  broadcastStrides(left_strides, right_strides, mask, left.strides(),
                   right.strides());

  return {S(shape, left_strides, left, 0, std::is_const<S>::value),
          T(shape, right_strides, right, 0, std::is_const<T>::value)};
}
template std::tuple<Tensor, Tensor> broadcast(Tensor &, Tensor &);
template std::tuple<const Tensor, Tensor> broadcast(const Tensor &, Tensor &);
template std::tuple<Tensor, const Tensor> broadcast(Tensor &, const Tensor &);
template std::tuple<const Tensor, const Tensor> broadcast(const Tensor &,
                                                          const Tensor &);

} // namespace gradstudent
