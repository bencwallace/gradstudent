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

namespace gradstudent {

/* OPERATORS */

/**
 * @brief Addition operator for tensors.
 *
 * Tensors must be broadcastable.
 *
 * @return The result of element-wise addition of the two tensors.
 * @throws std::invalid_argument If the tensors cannot be broadcasted.
 */
Tensor operator+(const Tensor &tensor1, const Tensor &tensor2);

/**
 * @brief Unary negation operator for tensors.
 * @param tensor The tensor to be negated.
 * @return The result of element-wise negation of the tensor.
 */
Tensor operator-(const Tensor &tensor);

/**
 * @brief Subtraction operator for tensors.
 *
 * Tensors must be broadcastable.
 *
 * @return The result of element-wise subtraction of the two tensors.
 * @throws std::invalid_argument If the tensors cannot be broadcasted.
 */
Tensor operator-(const Tensor &tensor1, const Tensor &tensor2);

/**
 * @brief Element-wise multiplication operator for tensors.
 *
 * Tensors must be broadcastable.
 *
 * @return The result of element-wise multiplication of the two tensors.
 * @throws std::invalid_argument If the tensors cannot be broadcasted.
 */
Tensor operator*(const Tensor &tensor1, const Tensor &tensor2);

/**
 * @brief Equality operator for tensors.
 * @return True if the two tensors have the same shape and are element-wise
 * equal, false otherwise.
 */
bool operator==(const Tensor &tensor1, const Tensor &tensor2);

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
Tensor dot(const Tensor &tensor1, const Tensor &tensor2);

/**
 * @brief Computes the squared L2 norm of a tensor.
 * @param tensor The tensor.
 * @return A scalar tensor containing the squared L2 norm of the tensor.
 */
Tensor norm2(const Tensor &tensor);

/* CONVOLUTIONS */

/**
 * @brief Computes the convolution of an input tensor with a kernel.
 *
 * The input tensor and kernel must have the same rank.
 *
 * @param input The input tensor
 * @param kernel The kernel tensor
 * @return Tensor
 * @todo Support broadcasting
 * @todo Support padding
 */
Tensor conv(const Tensor &input, const Tensor &kernel);

/* VIEWS */

/**
 * @brief Flattens a tensor.
 *
 * Produces a view of the original tensor with a single dimension containing all
 * the elements of the original tensor.
 *
 * @param tensor The tensor to be flattened.
 * @return The flattened tensor.
 */
Tensor flatten(Tensor &tensor);

/**
 * @overload
 *
 */
const Tensor flatten(const Tensor &tensor);

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
Tensor slice(Tensor &tensor, const array_t &indices);

/**
 * @overload
 */
const Tensor slice(const Tensor &tensor, const array_t &indices);

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
template <typename S, typename T>
std::tuple<S, T> broadcast(S &tensor1, T &tensor2);

} // namespace gradstudent
