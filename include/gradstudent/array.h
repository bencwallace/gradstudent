/**
 * @file array.h
 * @author Ben Wallace (me@bcwallace.com)
 * @brief Convenience type for tensor indexing
 * @version 0.1
 * @date 2024-03-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <ostream>
#include <vector>

#include <boost/iterator/iterator_facade.hpp>

namespace gs {

using std::size_t;

class Array;

/** @brief Convenience definition for backwards-compatibility with old array_t definition */
using array_t = Array;

/**
 * @brief A sequence of numbers
 *
 * Suitable for indexing tensors. Basically a size-aware, dynamically-allocated
 * (C/C++) array. In particular, an initialized Array has a fixed size (an uninitialized
 * array may be resized once on initialization).
 */
class Array {
  struct sentinel {}; // needed to disambiguate constructor
  Array(size_t size, sentinel)
      : size_(size), data_(new size_t[size_ * sizeof(size_t)]){};

  class Iterator
      : public boost::iterator_facade<Iterator, size_t,
                                      std::random_access_iterator_tag> {
  public:
    using value_type = size_t;

    Iterator() : data_(nullptr) {}
    Iterator(size_t *data) : data_(data) {}

  private:
    size_t *data_;

    friend class boost::iterator_core_access;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    void increment() { ++data_; }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    void decrement() { --data_; }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    void advance(difference_type n) { data_ += n; }

    size_t distance_to(const Iterator &other) const {
      return other.data_ - data_;
    }

    bool equal(const Iterator &other) const { return data_ == other.data_; }

    size_t &dereference() const { return *data_; }
  };

public:
  /** @brief Array element type */
  using value_type = size_t;

  /**
   * @brief Uninitialized Array constructor.
   *
   * Useful when an array must be declared but its size is not yet known.
   * An uninitialized array may be initialized once by assignment.
   *
   */
  Array();

  /** @brief Array copy constructor */
  Array(const Array &other);

  /** @brief Array initializer list constructor */
  Array(std::initializer_list<size_t> data);

  /** @brief Array constructor from vector */
  Array(const std::vector<size_t> &data);

  /**
   * @brief Array fill constructor
   *
   * Constructs an array consisting of the given value repeated the given number of times.
   *
   * @param size Number of times to repeat the given value
   * @param value Value with which to fill the array
   */
  Array(size_t size, size_t value);

  ~Array() = default;

  /**
   * @brief Array copy assignment operator
   *
   * With the exception of uninitialized arrays, to which any array can be assigned,
   * an array can only be assigned another array of the same size.
   *
   * @param other 
   * @return Array& 
   */
  Array &operator=(const Array &other);

  /**
   * @brief Array equality operator
   *
   * @param other 
   * @return true If the other array has the same size and values
   * @return false Otherwise
   */
  bool operator==(const Array &other) const;

  /**
   * @brief Array non-equality operator
   *
   * @param other 
   * @return true 
   * @return false 
   */
  bool operator!=(const Array &other) const;

  /**
   * @brief Array subscript operator
   *
   * @param i 
   * @return size_t& The element at index i
   */
  size_t &operator[](size_t i) { return data_[i]; }

  /** @overload */
  size_t operator[](size_t i) const { return data_[i]; }

  /** @brief Array size */
  size_t size() const { return size_; }

  /** @brief Whether the Array is empty */
  bool empty() const { return size_ == 0; }

  /**
   * @brief Slices the Array
   *
   * Note that this returns a new Array containing a copy of the sliced elements,
   * not a view of the original Array.
   *
   * @param start 
   * @param stop 
   * @return array_t A new Array containing a copy of the elements from start (inclusive)
   * to stop (exclusive).
   */
  array_t slice(size_t start, size_t stop) const;

  /**
   * @brief Slices the Array to the end
   *
   * @param start 
   * @return array_t 
   */
  array_t sliceFrom(size_t start) const;

  /**
   * @brief Slices the array from the beginning.
   *
   * @param stop 
   * @return array_t 
   */
  array_t sliceTo(size_t stop) const;

  /** @brief Returns an iterator pointing to the start of the Array */
  auto begin() const { return Iterator(data_.get()); }
  /** @overload */
  auto begin() { return Iterator(data_.get()); }

  /** @brief Returns an iterator pointing one past the end of the Array */
  auto end() const { return Iterator(data_.get() + size_); }
  /** @overload */
  auto end() { return Iterator(data_.get() + size_); }

  /** @brief Returns a reverse iterator pointing to the end of the Array */
  auto rbegin() const {
    return std::reverse_iterator(Iterator(data_.get() + size_));
  }

private:
  size_t size_;
  std::unique_ptr<size_t[]> data_;
};

/** @brief Concatenation operator */
array_t operator|(const array_t &lhs, const array_t &rhs);

/**
 * @brief Scalar addition operator
 *
 * Adds rhs to each element of lhs
 *
 * @param lhs 
 * @param rhs 
 * @return array_t
 */
array_t operator+(const array_t &lhs, size_t rhs);

/** @brief Elementwise addition operator */
array_t operator+(const array_t &lhs, const array_t &rhs);

/**
 * @brief Elementwise subtraction operator
 *
 * Fails if subtraction would result in a negative element.
 *
 * @param lhs 
 * @param rhs 
 * @return array_t
 */
array_t operator-(const array_t &lhs, const array_t &rhs);

/** @brief Elementwise multiplication operator */
array_t operator*(const array_t &lhs, const array_t &rhs);

/**
 * @brief Elementwise division operator
 *
 * Fails if division would result in a non-integer element.
 *
 * @param lhs 
 * @param rhs 
 * @return array_t 
 */
array_t operator/(const array_t &lhs, const array_t &rhs);

/**
 * @brief Scalar division operator
 *
 * Fails if division would result in a non-integer element.
 *
 * @param lhs 
 * @param rhs 
 * @return array_t 
 */
array_t operator/(const array_t &lhs, size_t rhs);

/**
 * @brief Output stream operator
 *
 * Streams an array as a parentheses-enclosed, comma-separated list of its elements.
 *
 * @param os 
 * @param array 
 * @return std::ostream& 
 */
std::ostream &operator<<(std::ostream &os, const array_t &array);

} // namespace gs
