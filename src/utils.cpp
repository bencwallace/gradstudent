#include <cstring>

#include "utils.h"

Array::Array(const Array &array)
    : size(array.size), data(std::make_unique<size_t[]>(size)) {
  std::memcpy(data.get(), array.data.get(), size * sizeof(size_t));
}

Array::Array(size_t size)
    : size(size), data(std::make_unique<size_t[]>(size)) {}

Array::Array(std::initializer_list<size_t> data)
    : size(data.size()), data(std::make_unique<size_t[]>(size)) {
  std::copy(data.begin(), data.end(), this->data.get());
}

size_t Array::operator[](size_t i) const { return data[i]; }

size_t &Array::operator[](size_t i) { return data[i]; }
