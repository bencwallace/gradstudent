/**
 * @file array.h
 * @author Ben Wallace (me@bcwallace.com)
 * @brief Simple type definitions
 * @version 0.1
 * @date 2024-03-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace gs {

using std::size_t;

/**
 * @brief A sequence of numbers
 *
 * Suitable for indexing tensors
 */
class Array {
  using array_t = Array;
  using iterator = std::vector<size_t>::const_iterator;

public:
  using value_type = size_t;

  Array() = default;
  Array(const Array &) = default;
  Array(std::initializer_list<size_t> data) : data_(data) {}
  Array(const iterator &begin, const iterator &end) : data_(begin, end) {}
  Array(size_t size, size_t value) : data_(size, value) {}
  ~Array() = default;

  Array &operator=(const Array &other) {
    data_ = other.data_;
    return *this;
  }

  bool operator==(const Array &other) const { return data_ == other.data_; }
  bool operator!=(const Array &other) const { return data_ != other.data_; }

  size_t &operator[](size_t i) { return data_[i]; }
  size_t operator[](size_t i) const { return data_[i]; }

  size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto rbegin() const { return data_.rbegin(); }

private:
  std::vector<size_t> data_;
};

using array_t = Array;

// @cond
void checkEqualSize(const array_t &lhs, const array_t &rhs);

array_t operator|(const array_t &lhs, const array_t &rhs);

array_t operator+(const array_t &lhs, size_t rhs);

array_t operator+(const array_t &lhs, const array_t &rhs);

array_t operator-(const array_t &lhs, const array_t &rhs);

array_t operator*(const array_t &lhs, const array_t &rhs);

array_t operator/(const array_t &lhs, const array_t &rhs);

array_t operator/(const array_t &lhs, size_t rhs);

array_t slice(const array_t &array, size_t start, size_t stop);

array_t sliceFrom(const array_t &array, size_t start);

array_t sliceTo(const array_t &array_t, size_t stop);

std::ostream &operator<<(std::ostream &os, const array_t &array);
// @endcond

} // namespace gs
