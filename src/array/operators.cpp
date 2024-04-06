#include <sstream>

#include "array.h"

namespace gs {

Array &Array::operator=(const Array &other) {
  if (this == &other) {
    return *this;
  }
  if (data_ == nullptr) {
    size_ = other.size_;
    data_ = std::make_unique<size_t[]>(size_);
  }
  if (other.size_ != size_) {
    std::stringstream ss;
    ss << "Cannot assign array of size " << other.size_ << " to array of size "
       << size_;
  }
  std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(size_t));
  return *this;
}

bool Array::operator==(const Array &other) const {
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
bool Array::operator!=(const Array &other) const { return !(*this == other); }

void checkEqualSize(const array_t &lhs, const array_t &rhs) {
  if (lhs.size() != rhs.size()) {
    std::stringstream ss;
    ss << "Expected equal size arrays, got sizes " << lhs.size() << " and "
       << rhs.size();
    throw std::invalid_argument(ss.str());
  }
}

array_t operator|(const array_t &lhs, const array_t &rhs) {
  array_t result(lhs.size() + rhs.size(), 0);
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
  array_t result(lhs.size(), 0);
  for (size_t i = 0; i < lhs.size(); ++i) {
    result[i] = lhs[i] + rhs[i];
  }
  return result;
}

array_t operator-(const array_t &lhs, const array_t &rhs) {
  checkEqualSize(lhs, rhs);
  array_t result(lhs.size(), 0);
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
  array_t result(lhs.size(), 0);
  for (size_t i = 0; i < lhs.size(); ++i) {
    result[i] = lhs[i] * rhs[i];
  }
  return result;
}

array_t operator/(const array_t &lhs, const array_t &rhs) {
  checkEqualSize(lhs, rhs);
  array_t result(lhs.size(), 0);
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

std::ostream &operator<<(std::ostream &os, const array_t &array) {
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

} // namespace gs
