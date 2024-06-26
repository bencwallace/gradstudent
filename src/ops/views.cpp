#include <sstream>

#include "gradstudent/internal/utils.h"
#include "gradstudent/ops.h"
#include "gradstudent/tensor.h"

namespace gs {

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

  array_t result_shape(tensor.ndims(), 0);
  array_t result_strides(tensor.ndims(), 0);
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
  return Tensor(result_shape, result_strides, tensor, tensor.offset(),
                tensor.ro());
}

// NOLINTNEXTLINE(readability-const-return-type)
const Tensor permute(const Tensor &tensor, std::initializer_list<size_t> axes) {
  auto [result_shape, result_strides] = permuteCommon(tensor, axes);
  return Tensor(result_shape, result_strides, tensor, tensor.offset(), true);
}

// TRUNCATE

array_t truncateShape(const Tensor &tensor, const array_t &start,
                      const array_t &stop) {
  if (start.size() != stop.size()) {
    std::stringstream ss;
    ss << "Expected start and stop arrays of equal length, got " << start.size()
       << " and " << stop.size();
    throw std::invalid_argument(ss.str());
  }
  if (start.size() > tensor.ndims()) {
    std::stringstream ss;
    ss << "Start and stop arrays have size " << start.size()
       << ", exceeding tensor rank " << tensor.ndims();
    throw std::invalid_argument(ss.str());
  }

  array_t Start = start | array_t(tensor.ndims() - start.size(), 0);
  array_t Stop = stop | tensor.shape().sliceFrom(start.size());
  array_t result_shape;
  try {
    result_shape = Stop - Start;
  } catch (const std::invalid_argument &e) {
    std::stringstream ss;
    ss << "Start index must precede stop index. Got " << start << " and "
       << stop;
    throw std::invalid_argument(ss.str());
  }

  return result_shape;
}

Tensor truncate(Tensor &tensor, const array_t &start, const array_t &stop) {
  auto result_shape = truncateShape(tensor, start, stop);
  return Tensor(result_shape, tensor.strides(), tensor, tensor.toIndex(start),
                tensor.ro());
}

// NOLINTNEXTLINE(readability-const-return-type)
const Tensor truncate(const Tensor &tensor, const array_t &start,
                      const array_t &stop) {
  auto result_shape = truncateShape(tensor, start, stop);
  return Tensor(result_shape, tensor.strides(), tensor, tensor.toIndex(start),
                true);
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

  auto result_shape = tensor.shape().sliceFrom(mIdx.size());
  auto result_strides = tensor.strides().sliceFrom(mIdx.size());
  return {result_shape, result_strides};
}

Tensor slice(Tensor &tensor, const array_t &mIdx) {
  auto [result_shape, result_strides] = sliceCommon(tensor, mIdx);
  return Tensor(result_shape, result_strides, tensor, tensor.toIndex(mIdx),
                tensor.ro());
}

// NOLINTNEXTLINE(readability-const-return-type)
const Tensor slice(const Tensor &tensor, const array_t &mIdx) {
  auto [result_shape, result_strides] = sliceCommon(tensor, mIdx);
  return Tensor(result_shape, result_strides, tensor, tensor.toIndex(mIdx),
                true);
}

// BROADCAST

void broadcastStrides(array_t &out, const std::vector<int> &mask,
                      const array_t &in, int side) {
  out = in | array_t(mask.size() - out.size(), 1); // TODO: something's off here
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

// NOLINTNEXTLINE(readability-const-return-type)
template <typename T> T broadcast(T &tensor, const array_t &shape) {
  array_t out_shape;
  array_t out_strides;
  auto mask = broadcastShapes(out_shape, tensor.shape(), shape);
  broadcastStrides(out_strides, mask, tensor.strides(), BCAST_LEFT);
  return Tensor(out_shape, out_strides, tensor, tensor.offset(),
                std::is_const_v<T> || tensor.ro());
}
template Tensor broadcast<Tensor>(Tensor &, const array_t &);
template const Tensor broadcast<const Tensor>(const Tensor &, const array_t &);

template <typename S, typename T>
std::tuple<S, T> broadcast(S &left, T &right) {
  array_t shape;
  array_t left_strides;
  array_t right_strides;
  auto mask = broadcastShapes(shape, left.shape(), right.shape());
  broadcastStrides(left_strides, right_strides, mask, left.strides(),
                   right.strides());

  return {S(shape, left_strides, left, left.offset(),
            std::is_const_v<S> || left.ro()),
          T(shape, right_strides, right, right.offset(),
            std::is_const_v<T> || right.ro())};
}
template std::tuple<Tensor, Tensor> broadcast(Tensor &, Tensor &);
template std::tuple<const Tensor, Tensor> broadcast(const Tensor &, Tensor &);
template std::tuple<Tensor, const Tensor> broadcast(Tensor &, const Tensor &);
template std::tuple<const Tensor, const Tensor> broadcast(const Tensor &,
                                                          const Tensor &);

void Tensor::reshapeCommon(const array_t &shape, const array_t &strides) const {
  if (prod(shape) != size()) {
    std::stringstream ss;
    ss << "Cannot reshape tensor of size " << size() << " to shape " << shape;
    throw std::invalid_argument(ss.str());
  }
  if (strides.size() != shape.size()) {
    std::stringstream ss;
    ss << "Expected strides of size " << shape.size() << ", got "
       << strides.size();
    throw std::invalid_argument(ss.str());
  }
}

Tensor Tensor::reshape(const array_t &shape, const array_t &strides) {
  reshapeCommon(shape, strides);
  return Tensor(shape, strides, *this, offset_, ro_);
}

// NOLINTNEXTLINE(readability-const-return-type)
const Tensor Tensor::reshape(const array_t &shape,
                             const array_t &strides) const {
  reshapeCommon(shape, strides);
  return Tensor(shape, strides, *this, offset_, true);
}

Tensor Tensor::reshape(const array_t &shape) {
  return reshape(shape, defaultStrides(shape));
}

// NOLINTNEXTLINE(readability-const-return-type)
const Tensor Tensor::reshape(const array_t &shape) const {
  return reshape(shape, defaultStrides(shape));
}

} // namespace gs
