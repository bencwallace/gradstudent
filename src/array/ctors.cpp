#include "array.h"
#include <cstring>

namespace gs {

Array::Array() : size_(0), data_(nullptr){};

Array::Array(const Array &other) : Array(other.size_, sentinel{}) {
  std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(size_t));
}

Array::Array(std::initializer_list<size_t> data)
    : Array(std::vector<size_t>(data)) {}

Array::Array(const std::vector<size_t> &data) : Array(data.size(), sentinel{}) {
  std::memcpy(data_.get(), data.data(), size_ * sizeof(size_t));
}

Array::Array(size_t size, size_t value) : Array(size, sentinel{}) {
  for (size_t i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

} // namespace gs
