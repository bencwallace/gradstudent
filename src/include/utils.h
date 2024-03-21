/**
 * @file utils.h
 * @author Ben Wallace (me@bcwallace.com)
 * @brief Additional utility functions
 * @version 0.1
 * @date 2024-03-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <ostream>
#include <vector>

#include "types.h"

namespace gradstudent {

class Tensor;

enum { BCAST_LEFT = -1, BCAST_NONE = 0, BCAST_RIGHT = 1 };

/* ARITHMETIC */

/**
 * @brief Computes default strides for a given shape
 *
 * Strides are computed in row-major order, i.e. each stride is the product of
 * the following dimensions. In particular, the final stride is 1 (the empty
 * product).
 */
array_t defaultStrides(const array_t &shape);

// @cond
template <typename P, typename Q>
size_t sumProd(const P &left, const Q &right, size_t start, size_t end) {
  size_t sum = 0;
  for (size_t i = start; i < end; ++i) {
    sum += left[i] * right[i];
  }
  return sum;
}

template <typename P, typename Q>
size_t sumProd(const P &left, const Q &right) {
  return sumProd(left, right, 0, std::min(left.size(), right.size()));
}

template <typename T> size_t prod(const T &array) {
  size_t result = 1;
  for (size_t i = 0; i < array.size(); ++i) {
    result *= array[i];
  }
  return result;
}
// @endcond

/**
 * @brief Broadcasts two shapes to a common shape.
 *
 * The output parameter is set to the broadcasted shape.
 *
 * The return value is a vector
 * of enum values BCAST_LEFT, BCAST_NONE, and BCAST_RIGHT, indicating which
 * shape (left, right, or neither) must be broadcasted along the corresponding
 * dimension. This vector can be passed along to the broadcastStrides function
 * to obtain the strides of tensors broadcasted with the given shapes. This is
 * mostly for internal use.
 *
 * @param[out] out Broadcasted shape.
 * @param left Left shape.
 * @param right Right shape.
 * @return std::vector<int> Vector indicating which shape is broadcasted along
 * the corresponding dimension.
 * @throw std::invalid_argument If the shapes are not broadcastable.
 */
std::vector<int> broadcastShapes(array_t &out, const array_t &left,
                                 const array_t &right);

// @cond
void checkCompatibleShape(const Tensor &, const Tensor &);

/* STREAMS */

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &array) {
  std::ostream &result = os << "(";
  for (int i = 0; i < (int)array.size() - 1; ++i) {
    result << array[i] << ", ";
  }
  if (!array.empty()) {
    result << array[array.size() - 1];
  }
  if (array.size() == 1) {
    result << ",";
  }
  result << ")";
  return result;
}
// @endcond

} // namespace gradstudent
