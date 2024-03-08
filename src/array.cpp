#include <cstring>
#include <sstream>

#include "array.h"

Array::Array(const Array &array)
    : data(std::make_unique<size_t[]>(array.size)), size(array.size) {
  std::memcpy(data.get(), array.data.get(), size * sizeof(size_t));
}

Array::Array(size_t size)
    : data(std::make_unique<size_t[]>(size)), size(size) {}

Array::Array(std::initializer_list<size_t> data)
    : data(std::make_unique<size_t[]>(data.size())), size(data.size()) {
  std::copy(data.begin(), data.end(), this->data.get());
}

Array Array::operator=(const Array &other) {
  if (size != other.size) {
    std::stringstream ss;
    ss << "Expected arrays of equal size, got " << size << " and "
       << other.size;
    throw std::invalid_argument(ss.str());
  }
  std::copy(other.data.get(), other.data.get() + other.size, data.get());
  return *this;
}

size_t Array::operator[](size_t i) const { return data[i]; }

size_t &Array::operator[](size_t i) { return data[i]; }

bool Array::operator!=(const Array &other) const {
  if (size != other.size) {
    return false;
  }
  for (size_t i = 0; i < size; ++i) {
    if ((*this)[i] != other[i]) {
      return false;
    }
  }
  return true;
}

size_t Array::prod() const {
  size_t result = 1;
  for (size_t i = 0; i < size; ++i) {
    result *= data[i];
  }
  return result;
}

std::ostream &operator<<(std::ostream &os, Array const &array) {
  std::ostream &result = os << "(";
  for (int i = 0; i < (int) array.size - 1; ++i) {
    result << array[i] << ", ";
  }
  if (array.size > 0) {
    result << array[array.size - 1];
  }
  if (array.size == 1) {
    result << ",";
  }
  result << ")";
  return result;
}

Array zerosArray(size_t size) {
  Array result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = 0;
  }
  return result;
}
