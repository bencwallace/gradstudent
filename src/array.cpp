#include <sstream>

#include "array.h"

namespace gs {

void checkEqualSize(const array_t &lhs, const array_t &rhs) {
  if (lhs.size() != rhs.size()) {
    std::stringstream ss;
    ss << "Expected equal size arrays, got sizes " << lhs.size() << " and "
       << rhs.size();
    throw std::invalid_argument(ss.str());
  }
}

array_t operator|(const array_t &lhs, const array_t &rhs) {
  array_t result(lhs.size() + rhs.size());
  std::copy(lhs.begin(), lhs.end(), result.begin());
  std::copy(rhs.begin(), rhs.end(), result.begin() + lhs.size());
  return result;
}

array_t operator+(const array_t &lhs, size_t rhs) {
  array_t result(lhs);
  for (size_t &x : result) {
    x += rhs;
  }
  return result;
}

array_t operator+(const array_t &lhs, const array_t &rhs) {
  checkEqualSize(lhs, rhs);
  array_t result(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    result[i] = lhs[i] + rhs[i];
  }
  return result;
}

array_t operator-(const array_t &lhs, const array_t &rhs) {
  checkEqualSize(lhs, rhs);
  array_t result(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] < rhs[i]) {
      throw std::invalid_argument("Subtraction resulted in negative value");
    }
    result[i] = lhs[i] - rhs[i];
  }
  return result;
}

array_t operator*(const array_t &lhs, const array_t &rhs) {
  checkEqualSize(lhs, rhs);
  array_t result(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    result[i] = lhs[i] * rhs[i];
  }
  return result;
}

array_t operator/(const array_t &lhs, const array_t &rhs) {
  checkEqualSize(lhs, rhs);
  array_t result(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] % rhs[i] != 0) {
      throw std::invalid_argument("Division resulted in non-integer value");
    }
    result[i] = lhs[i] / rhs[i];
  }
  return result;
}

array_t operator/(const array_t &lhs, size_t rhs) {
  array_t result(lhs);
  for (size_t &x : result) {
    x /= rhs;
  }
  return result;
}

array_t slice(const array_t &array, size_t start, size_t stop) {
  return array_t(array.begin() + start, array.begin() + stop);
}

array_t sliceFrom(const array_t &array, size_t start) {
  return slice(array, start, array.size());
}

array_t sliceTo(const array_t &array, size_t stop) {
  return slice(array, 0, stop);
}

} // namespace gs
