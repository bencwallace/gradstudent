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

#include <initializer_list>
#include <memory>
#include <numeric>

#include "array.h"
#include "internal/utils.h"

namespace gradstudent {

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
  std::shared_ptr<double[]> data_; // NOLINT(cppcoreguidelines-avoid-c-arrays)

  void ensureWritable();

  void assignOther(const Tensor &);
  void assignSelf(const Tensor &);

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
   * @brief Tensor convenience constructor.
   *
   * Constructs a tensor with the given shape, strides, and data contents.
   */
  Tensor(const array_t &shape, const array_t &strides,
         const std::vector<double> &data);

  /**
   * @brief Tensor convenience constructor.
   *
   * Constructs a tensor with the given shape and data contents and default
   * strides.
   */
  Tensor(const array_t &shape, const std::vector<double> &data);

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

  // @cond
  inline size_t toIndex(const array_t &mIdx) const {
    return offset_ +
           std::inner_product(mIdx.begin(), mIdx.end(), strides_.begin(), 0);
  }
  // @endcond

  /* GETTERS/SETTERS */

  /**
   * @brief Returns the tensor size (number of elements)
   */
  inline size_t size() const { return size_; }

  /**
   * @brief Returns the tensor rank (number of dimensions)
   */
  inline size_t ndims() const { return shape_.size(); }

  /**
   * @brief Returns the tensor shape
   */
  inline const array_t &shape() const { return shape_; }

  /**
   * @brief Returns the tensor strides
   */
  inline const array_t &strides() const { return strides_; }

  /* FRIEND OPERATORS */

  friend Tensor operator+(const Tensor &, const Tensor &);
  friend Tensor operator-(const Tensor &);
  friend Tensor operator-(const Tensor &, const Tensor &);
  friend Tensor operator*(const Tensor &, const Tensor &);
  friend bool operator==(const Tensor &, const Tensor &);
};

} // namespace gradstudent
