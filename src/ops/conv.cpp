#include <functional>
#include <sstream>

#include "ops.h"
#include "tensor_iter.h"

namespace gradstudent {

void slidingWindowTransform(
    Tensor &result, const Tensor &input,
    std::function<Tensor(const Tensor &, const array_t &)> windowFn,
    std::function<double(const Tensor &)> transform) {
  for (auto [resIdx, res] : ITensorIter(result)) {
    Tensor window(windowFn(input, resIdx));
    res = transform(window);
  }
}

void slidingWindowTransformNoStride(Tensor &result, const Tensor &input,
                                    const array_t &windowShape,
                                    std::function<double(Tensor)> transform) {
  slidingWindowTransform(
      result, input,
      [&](const Tensor &input, const array_t &resIdx) {
        return truncate(input, resIdx, resIdx + windowShape);
      },
      transform);
}

void slidingWindowTransformFullStride(Tensor &result, const Tensor &input,
                                      const array_t &windowShape,
                                      std::function<double(Tensor)> transform) {
  slidingWindowTransform(
      result, input,
      [&](const Tensor &input, const array_t &resIdx) {
        return truncate(input, resIdx * windowShape,
                        resIdx * windowShape + windowShape);
      },
      transform);
}

Tensor singleConv(const Tensor &input, const Tensor &kernel) {
  array_t offset = kernel.shape() / 2;
  array_t result_shape =
      input.shape() - kernel.shape() + array_t(kernel.ndims(), 1);

  Tensor result(result_shape);
  slidingWindowTransformNoStride(
      result, input, kernel.shape(),
      [&](const Tensor &window) { return sum(window * kernel); });

  return result;
}

Tensor conv(const Tensor &input, const Tensor &kernel) {
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

  if (kernel.ndims() == input.ndims()) {
    return singleConv(input, kernel);
  }

  array_t singleKernelShape(kernel.shape().begin() + 1, kernel.shape().end());
  array_t singleResultShape =
      input.shape() - singleKernelShape + array_t(input.ndims(), 1);
  array_t resultShape(kernel.ndims());
  resultShape[0] = kernel.shape()[0];
  std::copy(singleResultShape.begin(), singleResultShape.end(),
            resultShape.begin() + 1);

  Tensor result(resultShape);
  for (size_t i = 0; i < kernel.shape()[0]; ++i) {
    slice(result, {i}) = singleConv(input, slice(kernel, array_t{i}));
  }
  return result;
}

Tensor maxPool(const Tensor &input, const array_t &poolShape) {
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

} // namespace gradstudent
