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
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <ostream>
#include <vector>

#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/reverse_iterator.hpp>

namespace gs {

using std::size_t;

/**
 * @brief A sequence of numbers
 *
 * Suitable for indexing tensors
 */
class Array {
  using array_t = Array;

  struct sentinel {}; // needed to disambiguate constructor
  Array(size_t size, sentinel)
      : size_(size), data_(new size_t[size_ * sizeof(size_t)]){};

public:
  using value_type = size_t;

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
    void increment() { ++data_; }
    void decrement() { --data_; }
    void advance(size_t n) { data_ += n; }
    size_t distance_to(const Iterator &other) const {
      return other.data_ - data_;
    }
    bool equal(const Iterator &other) const { return data_ == other.data_; }
    size_t &dereference() const { return *data_; }
  };

  Array();

  Array(const Array &other);

  Array(std::initializer_list<size_t> data);

  Array(const std::vector<size_t> &data);

  Array(size_t size, size_t value);

  ~Array() = default;

  Array &operator=(const Array &other);

  bool operator==(const Array &other) const;

  bool operator!=(const Array &other) const;

  size_t &operator[](size_t i) { return data_[i]; }
  size_t operator[](size_t i) const { return data_[i]; }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  array_t slice(size_t start, size_t stop) const;
  array_t sliceFrom(size_t start) const;
  array_t sliceTo(size_t stop) const;

  auto begin() const { return Iterator(data_.get()); }
  auto end() const { return Iterator(data_.get() + size_); }
  auto begin() { return Iterator(data_.get()); }
  auto end() { return Iterator(data_.get() + size_); }
  auto rbegin() const {
    return std::reverse_iterator(Iterator(data_.get() + size_));
  }

private:
  size_t size_;
  std::unique_ptr<size_t[]> data_;
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

std::ostream &operator<<(std::ostream &os, const array_t &array);
// @endcond

} // namespace gs
