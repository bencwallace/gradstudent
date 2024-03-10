#pragma once

#include <initializer_list>
#include <memory>

#include "array.h"
#include "tensor_data.h"

class Tensor {

private:
  size_t ndims_;
  Array shape_;
  Array strides_;
  std::shared_ptr<TensorData> data_;

  Tensor(const Array &, const Array &);

  size_t toIndex(const Array &) const;
  Array toMultiIndex(size_t) const;
  void checkCompatibleShape(const Tensor &) const;

public:
  Tensor(const Array &shape);

  Tensor(const Array &shape, std::initializer_list<double> data);
  Tensor(double);

  Tensor(const Array &shape, const Array &strides, const Tensor &);

  double operator[](size_t) const;
  double &operator[](size_t);
  double operator[](const Array &) const;
  double &operator[](const Array &);
  Tensor operator+(const Tensor &) const;
  Tensor operator-() const;
  Tensor operator-(const Tensor &) const;
  Tensor operator*(const Tensor &) const;

  size_t toIndex(const Array &, size_t, size_t) const;
  size_t size() const;
  size_t ndims() const;
  const Array &shape() const;
  const Array &strides() const;

  friend Tensor operator*(double, const Tensor &);
};

Tensor operator*(double, const Tensor &);
