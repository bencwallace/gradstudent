#include <cstring>

#include "utils.h"

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

std::ostream &operator<<(std::ostream &os, Array const &array) {
  std::ostream &result = os << "(";
  for (size_t i = 0; i < array.size - 1; ++i) {
    result << array[i] << ", ";
  }
  result << array[array.size - 1];
  if (array.size == 1) {
    result << ",";
  }
  result << ")";
  return result;
}
