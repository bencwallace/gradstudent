/**
 * @file ops.h
 * @author Ben Wallace (me@bcwallace.com)
 * @brief Operations on tensors
 * @version 0.1
 * @date 2024-03-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <initializer_list>
#include <tuple>

#include "array.h"
#include "tensor.h"

namespace gs {

/* OPERATORS */

/* ACTIVATIONS */

Tensor relu(const Tensor &tensor);

/* REDUCTIONS */

size_t argmax(const Tensor &tensor);

double max(const Tensor &tensor);

double sum(const Tensor &tensor);

/* LINEAR ALGEBRA */

/**
 * @brief Computes the dot product of two tensors.
 *
 * Computes a contraction of the of the first tensor along its final axis with
 * the second tensor along its first axis.
 *
 * @return The dot product of the two tensors.
 * @throws std::invalid_argument If the tensor contraction axes have different
 * sizes.
 */
Tensor dot(const Tensor &left, const Tensor &right);

/**
 * @brief Computes the squared L2 norm of a tensor.
 * @param tensor The tensor.
 * @return A scalar tensor containing the squared L2 norm of the tensor.
 */
Tensor norm2(const Tensor &tensor);

/* SLIDING WINDOW TRANSFORMS */

/**
 * @brief Computes the convolution of an input tensor with a kernel.
 *
 * The kernel rank can be at most one greater than the input rank.
 * When it is strictly one greater, the first dimension specifies the
 * filter.
 *
 * Suppose the input has shape (m_1, ..., m_d) and the kernel has shape
 * (k_0, k_1, ..., k_d). Denote the case where input and kernel rank match
 * by allowing m_0 == 0. Then it is required that k_i == m_i for all
 * i = d-n+1, ..., d. These dimensions are contracted and the output will
 * have shape (k_0, p_1, ..., p_{d-n}) with p_j = m_j - k_j + 1 for each j.
 *
 * @param input The input tensor
 * @param kernel The kernel tensor
 * @param n The number of dimensions over which to perform the convolution.
 * @return Tensor
 * @todo Support broadcasting
 * @todo Support padding
 */
Tensor conv(const Tensor &input, const Tensor &kernel, size_t n = 0);

Tensor maxPool(const Tensor &input, const array_t &poolShape);

/* VIEWS */

/**
 * @brief Flattens a tensor.
 *
 * Produces a new tensor with a single dimension containing all
 * the elements of the original tensor.
 *
 * @param tensor The tensor to be flattened.
 * @return The flattened tensor.
 */
Tensor flatten(const Tensor &tensor);

/**
 * @brief Permutes the dimensions of a tensor.
 *
 * Produces a view of the original tensor with the dimensions permuted according
 * to the specified order.
 *
 * @param tensor The tensor to be permuted.
 * @param axes The new order of dimensions.
 * @return The permuted tensor.
 * @throws std::invalid_argument If the number of axes is different from the
 * tensor's rank.
 */
Tensor permute(Tensor &tensor, std::initializer_list<size_t> axes);

/**
 * @overload
 */
const Tensor permute(const Tensor &tensor, std::initializer_list<size_t> axes);

Tensor truncate(Tensor &tensor, const array_t &start, const array_t &stop);

const Tensor truncate(const Tensor &tensor, const array_t &start,
                      const array_t &stop);

/**
 * @brief Slices a tensor.
 *
 * Produces a view of the original tensor with the specified indices fixed and
 * all other indices free.
 *
 * @param tensor The tensor to be sliced.
 * @param indices The indices specifying the slice.
 * @return The sliced tensor.
 */
Tensor slice(Tensor &tensor, const array_t &mIdx);

/**
 * @overload
 */
const Tensor slice(const Tensor &tensor, const array_t &mIdx);

/**
 * @brief Broadcasts a tensor to a given shape.
 *
 * Produces a view of the original tensor with the given shape.
 *
 * @tparam T The type of the tensor (Tensor or const Tensor).
 * @param tensor The tensor to be broadcasted.
 * @param shape The shape to broadcast the tensor to.
 * @return The broadcasted tensor.
 * @throws std::invalid_argument If the tensor cannot be broadcasted to the
 * given shape.
 */
template <typename T> T broadcast(T &tensor, const array_t &shape);

/**
 * @brief Broadcasts two tensors to a common shape.
 *
 * Produces views of the original tensors with a common shape.
 *
 * @tparam S The type of the first tensor (Tensor or const Tensor).
 * @tparam T The type of the second tensor (Tensor or const Tensor).
 * @return A tuple containing the broadcasted tensors.
 * @throws std::invalid_argument If the tensors cannot be broadcasted.
 */
template <typename S, typename T> std::tuple<S, T> broadcast(S &left, T &right);

} // namespace gs
