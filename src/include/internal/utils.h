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

#include <numeric>
#include <vector>

#include "array.h"

namespace gs {

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
template <typename Iterable> size_t prod(const Iterable &iterable) {
  return std::reduce(iterable.begin(), iterable.end(), 1,
                     std::multiplies<typename Iterable::value_type>());
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
// @endcond

} // namespace gs
