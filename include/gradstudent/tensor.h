/**
 * @file tensor.h
 * @author Ben Wallace (me@bcwallace.com)
 * @brief Tensor class definition
 * @version 0.1
 * @date 2024-03-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <memory>
#include <numeric>

#include "gradstudent/array.h"

namespace gs {

/**
 * @class Tensor
 * @brief Represents a multi-dimensional array.
 *
 * The Tensor class provides functionality for creating, manipulating, and
 * accessing multi-dimensional arrays.
 */
class Tensor {

private:
  bool ro_ = false; // read-only (for views of const tensors)
  size_t offset_;
  const size_t size_;
  const array_t shape_;
  array_t strides_;
  std::shared_ptr<double[]> data_;

  void ensureWritable();

  void assignOther(const Tensor &);
  void assignSelf(const Tensor &);

  void reshapeCommon(const array_t &shape, const array_t &strides) const;

public:
  template <bool... Const> friend class TensorIter;

  /* CONSTRUCTORS */

  /**
   * @brief Tensor copy constructor
   *
   * The copied tensor will not be a view of the original tensor, i.e. it will
   * have a separate data buffer.
   */
  Tensor(const Tensor &);

  /**
   * @brief Tensor view constructor.
   *
   * Constructs a view of the given tensor, with the given shape, strides, and
   * offset. A view may be read-only, in which case a copy-on-write mechanism is
   * used when writing to the view, i.e. the original data is copied and the
   * view ceases to be a view if a write is attempted.
   */
  explicit Tensor(const array_t &shape, const array_t &strides, const Tensor &,
                  size_t offset = 0, bool ro = false);

  /**
   * @brief Empty tensor constructor.
   *
   * Constructs a tensor with the given shape, default strides, and an allocated
   * but uninitialized data buffer.
   */
  Tensor(const array_t &shape);

  /**
   * @brief Empty strided tensor constructor
   *
   * Constructs a tensor with the given shape and strides, and an allocated but
   * uninitialized data buffer.
   */
  Tensor(const array_t &shape, const array_t &strides);

  /**
   * @brief Scalar tensor constructor.
   *
   * Constructs a scalar tensor with the given value.
   */
  Tensor(double);

  /**
   * @brief Tensor fill constructor.
   *
   * Constructs a tensor filled with the given value.
   */
  static Tensor fill(const array_t &shape, const array_t &strides,
                     double value);

  /**
   * @overload
   */
  static Tensor fill(const array_t &shape, double value);

  /**
   * @brief Tensor range constructor
   *
   * Constructs the 1D tensor of values from 0 (inclusive) to stop (exclusive)
   * with step size 1.
   */
  static Tensor range(int stop);

  /**
   * @brief Tensor range constructor
   *
   * Constructs the 1D tensor of values from start (inclusive) to stop (exclusive)
   * with step size 1.
   *
   */
  static Tensor range(int start, int stop);

  /**
   * @brief Tensor range constructor
   *
   * Constructs the 1D tensor of values from start (inclusive) to stop (exclusive),
   * with the given step size.
   */
  static Tensor range(int start, int stop, int step);

  ~Tensor() = default;

  /* OPERATORS */

  /**
   * @brief Tensor assignment operator.
   *
   * Copies the contents of the given tensor's data buffer into this one's.
   * The copied tensor will not be a view of the original tensor, i.e. it will
   * have a separate data buffer.
   */
  Tensor &operator=(const Tensor &);

  /**
   * @brief Tensor subscript operator.
   *
   * Returns the value at the given index in the data buffer.
   * Does not perform bounds checking.
   */
  inline double operator[](size_t i) const { return data_[offset_ + i]; }

  /**
   * @overload
   *
   * Can be used to set the value at the given index in the data buffer.
   */
  inline double &operator[](size_t i) {
    ensureWritable();
    return data_[offset_ + i];
  }

  /**
   * @brief Tensor multi-index subscript operator.
   */
  inline double operator[](const array_t &mIdx) const {
    return data_[toIndex(mIdx)];
  }

  /**
   * @overload
   *
   * Can be used to set the value at the given multi-index.
   */
  inline double &operator[](const array_t &mIdx) {
    ensureWritable();
    return data_[toIndex(mIdx)];
  }

  /**
   * @brief Scalar cast operator.
   *
   * Can be used to cast a scalar tensor to its single value.
   */
  explicit operator double() const;

  /* UTILITIES */

  /** @brief Computes the buffer index corresponding to the given multi-index */
  inline size_t toIndex(const array_t &mIdx) const {
    return offset_ +
           std::inner_product(mIdx.begin(), mIdx.end(), strides_.begin(), 0UL);
  }

  /* GETTERS/SETTERS */

  /**
   * @brief Returns the tensor size (number of elements)
   */
  inline size_t size() const { return size_; }

  /**
   * @brief Returns the tensor rank (number of dimensions)
   */
  inline size_t ndims() const { return shape_.size(); }

  /** @brief Returns the tensor offset */
  inline size_t offset() const { return offset_; }

  /** @brief Returns the value of tensor read-only flag */
  inline bool ro() const { return ro_; }

  /* VIEWS */

  /**
   * @brief Reshapes the tensor
   *
   * Returns a view of the tensor with the given shape, whose corresponding
   * size must be the same as the tensor size.
   *
   * @param shape The new shape
   * @return Tensor
   */
  Tensor reshape(const array_t &shape);

  /**
   * @brief Returns a view of the tensor with the given shape and strides.
   * The shape and strides arrays must have the same size and the size corresponding
   * to the shape must be the same as the tensor size.
   * 
   * @param shape The new shape
   * @param strides The new strides
   * @return Tensor 
   */
  Tensor reshape(const array_t &shape, const array_t &strides);

  /** @overload */
  const Tensor reshape(const array_t &shape) const;

  /** @overload */
  const Tensor reshape(const array_t &shape, const array_t &strides) const;

  /**
   * @brief Returns the tensor shape
   */
  inline const array_t &shape() const { return shape_; }

  /**
   * @brief Returns the tensor strides
   */
  inline const array_t &strides() const { return strides_; }

  /* FRIEND OPERATORS */

  /**
   * @brief Addition operator for tensors.
   *
   * Tensors must be broadcastable.
   *
   * @return The result of element-wise addition of the two tensors.
   * @throws std::invalid_argument If the tensors cannot be broadcasted.
   */
  friend Tensor operator+(const Tensor &, const Tensor &);

  /**
   * @brief Unary negation operator for tensors.
   * @return The result of element-wise negation of the tensor.
   */
  friend Tensor operator-(const Tensor &);

  /**
   * @brief Subtraction operator for tensors.
   *
   * Tensors must be broadcastable.
   *
   * @return The result of element-wise subtraction of the two tensors.
   * @throws std::invalid_argument If the tensors cannot be broadcasted.
   */
  friend Tensor operator-(const Tensor &, const Tensor &);

  /**
   * @brief Element-wise multiplication operator for tensors.
   *
   * Tensors must be broadcastable.
   *
   * @return The result of element-wise multiplication of the two tensors.
   * @throws std::invalid_argument If the tensors cannot be broadcasted.
   */
  friend Tensor operator*(const Tensor &, const Tensor &);

  /**
   * @brief Equality operator for tensors.
   * @return True if the two tensors have the same shape and are element-wise
   * equal, false otherwise.
   */
  friend bool operator==(const Tensor &, const Tensor &);
};

} // namespace gs
