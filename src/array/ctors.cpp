#include "array.h"

namespace gs {

Array::Array() : size_(0), data_(nullptr){};
Array::Array(const Array &other) : Array(other.size_, sentinel{}) {
  std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(size_t));
}
Array::Array(std::initializer_list<size_t> data)
    : Array(data.size(), sentinel{}) {
  size_t i = 0;
  for (const auto &x : data) {
    data_[i++] = x;
  }
}
Array::Array(const std::vector<size_t>::const_iterator &begin,
             const std::vector<size_t>::const_iterator &end)
    : Array(end - begin, sentinel{}) {
  for (auto it = begin; it != end; ++it) {
    data_[it - begin] = *it;
  }
}
Array::Array(const Iterator &begin, const Iterator &end)
    : Array(end - begin, sentinel{}) {
  for (auto it = begin; it != end; ++it) {
    data_[it - begin] = *it;
  }
}
Array::Array(size_t size, size_t value) : Array(size, sentinel{}) {
  for (size_t i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

} // namespace gs
