#include "tensor_data.h"

/* TensorData */

TensorData::TensorData(size_t size) : size_(size) {}

TensorData::TensorData(const TensorData &other) : size_(other.size_) {}

size_t TensorData::size() const { return size_; }

/* TensorDataCpu */

TensorDataCpu::TensorDataCpu(size_t size)
    : TensorData(size), data(new double[size]) {}

double TensorDataCpu::operator[](size_t i) const { return data[i]; }

double &TensorDataCpu::operator[](size_t i) { return data[i]; }
