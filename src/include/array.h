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
#include <ostream>
#include <sstream>
#include <vector>

namespace gradstudent {

using std::size_t;

/**
 * @brief A sequence of numbers
 *
 * Suitable for indexing tensors
 */
using array_t = std::vector<size_t>;

// @cond
void checkEqualSize(const array_t &lhs, const array_t &rhs);

array_t operator+(const array_t &lhs, const array_t &rhs);

array_t operator-(const array_t &lhs, const array_t &rhs);

array_t operator/(const array_t &lhs, size_t rhs);

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
