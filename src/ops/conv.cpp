#include <functional>
#include <sstream>

#include "gradstudent/iter.h"
#include "gradstudent/ops.h"

namespace gs {

/* SLIDING WINDOW HELPER FUNCTIONS */

void slidingWindowTransform(
    Tensor &result, const Tensor &input,
    const std::function<Tensor(const Tensor &, const array_t &)> &windowFn,
    const std::function<double(const Tensor &)> &transform) {
  for (auto [resIdx, res] : ITensorIter(result)) {
    Tensor window(windowFn(input, resIdx));
    res = transform(window);
  }
}

void slidingWindowTransformNoStride(
    Tensor &result, const Tensor &input, const array_t &windowShape,
    const std::function<double(Tensor)> &transform) {
  slidingWindowTransform(
      result, input,
      [&](const Tensor &input, const array_t &resIdx) {
        return truncate(input, resIdx, resIdx + windowShape);
      },
      transform);
}

void slidingWindowTransformFullStride(
    Tensor &result, const Tensor &input, const array_t &windowShape,
    const std::function<double(Tensor)> &transform) {
  slidingWindowTransform(
      result, input,
      [&](const Tensor &input, const array_t &resIdx) {
        return truncate(input, resIdx * windowShape,
                        resIdx * windowShape + windowShape);
      },
      transform);
}

/* CONVOLUTION OVER ALL DIMENSIONS */

Tensor singleConv(const Tensor &input, const Tensor &kernel, size_t n) {
  array_t kernel_shape = kernel.shape().sliceTo(n);
  array_t result_shape = input.shape().sliceTo(n) - kernel_shape + 1;
  Tensor result(result_shape);
  slidingWindowTransformNoStride(
      result, input, kernel_shape,
      [&](const Tensor &window) { return sum(window * kernel); });

  return result;
}

Tensor multiConv(const Tensor &input, const Tensor &kernel, size_t n) {
  auto singleKernelShape = kernel.shape().slice(1, 1 + n);
  array_t singleResultShape = input.shape().sliceTo(n) - singleKernelShape + 1;
  auto resultShape = array_t{kernel.shape()[0]} | singleResultShape;

  Tensor result(resultShape);
  for (size_t i = 0; i < kernel.shape()[0]; ++i) {
    slice(result, {i}) = singleConv(input, slice(kernel, array_t{i}), n);
  }
  return result;
}

Tensor conv(const Tensor &input, const Tensor &kernel, size_t n) {
  if (n > input.ndims()) {
    std::stringstream ss;
    ss << "Convolution rank " << n << " exceeds input rank " << input.ndims();
    throw std::invalid_argument(ss.str());
  }
  if (kernel.ndims() < input.ndims()) {
    std::stringstream ss;
    ss << "Input rank should not exceed kernel rank, got " << input.ndims()
       << " and " << kernel.ndims();
    throw std::invalid_argument(ss.str());
  }
  if (kernel.ndims() > 1 + input.ndims()) {
    std::stringstream ss;
    ss << "Kernel rank should not exceed input rank by more than 1. Got kernel "
          "rank "
       << kernel.ndims() << " and input rank " << input.ndims();
    throw std::invalid_argument(ss.str());
  }

  n = n > 0 ? n : input.ndims();
  if (kernel.ndims() == input.ndims()) {
    return singleConv(input, kernel, n);
  }
  return multiConv(input, kernel, n);
}

/* MAX POOLING */

Tensor singleMaxPool(const Tensor &input, const array_t &poolShape) {
  array_t result_shape;
  try {
    result_shape = input.shape() / poolShape;
  } catch (const std::invalid_argument &e) {
    std::stringstream ss;
    ss << "Pool shape " << poolShape << " does not divide input shape "
       << input.shape();
    throw std::invalid_argument(ss.str());
  }
  Tensor result(result_shape);

  slidingWindowTransformFullStride(
      result, input, poolShape,
      [](const Tensor &window) { return max(window); });

  return result;
}

Tensor maxPool(const Tensor &input, const array_t &poolShape) {
  if (input.ndims() > poolShape.size() + 1) {
    std::stringstream ss;
    ss << "Input rank must be at most one greater than pool rank, got "
       << input.ndims() << " and " << poolShape.size();
    throw std::invalid_argument(ss.str());
  }
  if (poolShape.size() == input.ndims()) {
    return singleMaxPool(input, poolShape);
  }

  array_t inputSliceShape = input.shape().sliceFrom(1);
  array_t resultSliceShape;
  try {
    resultSliceShape = inputSliceShape / poolShape;
  } catch (const std::invalid_argument &e) {
    std::stringstream ss;
    ss << "Pool shape " << poolShape << " does not divide input shape "
       << input.shape();
    throw std::invalid_argument(ss.str());
  }
  array_t result_shape = array_t{input.shape()[0]} | resultSliceShape;
  Tensor result(result_shape);

  for (size_t i = 0; i < input.shape()[0]; ++i) {
    slice(result, {i}) = singleMaxPool(slice(input, {i}), poolShape);
  }
  return result;
}

} // namespace gs
