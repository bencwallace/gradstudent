#include "tensor_data.h"

TensorData::TensorData(size_t size)
    : size_(size), data(new double[size]) {}

TensorData::TensorData(const TensorData &other)
    : size_(other.size_), data(other.data) {}

size_t TensorData::size() const {
  return size_;
}

TensorData &TensorData::operator=(const TensorData &other) {
  this->data = other.data;
  return *this;
}

double TensorData::operator[](size_t i) const {
  return data[i];
}

double &TensorData::operator[](size_t i) {
  return data[i];
}
