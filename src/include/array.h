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
#include <sstream>
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

  Array() : size_(0), data_(nullptr){};
  Array(const Array &other) : Array(other.size_, sentinel{}) {
    std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(size_t));
  }
  Array(std::initializer_list<size_t> data) : Array(data.size(), sentinel{}) {
    size_t i = 0;
    for (const auto &x : data) {
      data_[i++] = x;
    }
  }
  Array(const std::vector<size_t>::const_iterator &begin,
        const std::vector<size_t>::const_iterator &end)
      : Array(end - begin, sentinel{}) {
    for (auto it = begin; it != end; ++it) {
      data_[it - begin] = *it;
    }
  }
  Array(const Iterator &begin, const Iterator &end)
      : Array(end - begin, sentinel{}) {
    for (auto it = begin; it != end; ++it) {
      data_[it - begin] = *it;
    }
  }
  Array(size_t size, size_t value) : Array(size, sentinel{}) {
    for (size_t i = 0; i < size_; ++i) {
      data_[i] = value;
    }
  }
  ~Array() = default;

  Array &operator=(const Array &other) {
    if (this == &other) {
      return *this;
    }
    if (data_ == nullptr) {
      size_ = other.size_;
      data_ = std::make_unique<size_t[]>(size_);
    }
    if (other.size_ != size_) {
      std::stringstream ss;
      ss << "Cannot assign array of size " << other.size_
         << " to array of size " << size_;
    }
    std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(size_t));
    return *this;
  }

  bool operator==(const Array &other) const {
    if (size_ != other.size_) {
      return false;
    }
    for (size_t i = 0; i < size_; ++i) {
      if (data_[i] != other.data_[i]) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const Array &other) const { return !(*this == other); }

  size_t &operator[](size_t i) { return data_[i]; }
  size_t operator[](size_t i) const { return data_[i]; }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

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

array_t slice(const array_t &array, size_t start, size_t stop);

array_t sliceFrom(const array_t &array, size_t start);

array_t sliceTo(const array_t &array_t, size_t stop);

std::ostream &operator<<(std::ostream &os, const array_t &array);
// @endcond

} // namespace gs
