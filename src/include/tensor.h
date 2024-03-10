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
  Array offset_;
  std::shared_ptr<TensorData> data_;

  Tensor(const Array &shape, const Array &strides);

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
  Tensor operator[](const Array &) const;
  double &operator[](const Array &);
  Tensor operator+(const Tensor &) const;
  Tensor operator-() const;
  Tensor operator-(const Tensor &) const;
  Tensor operator*(const Tensor &) const;
  explicit operator double() const;

  size_t toIndex(const Array &, size_t, size_t) const;

  size_t size() const;
  size_t ndims() const;
  const Array &shape() const;
  const Array &strides() const;

  friend Tensor operator*(double, const Tensor &);
};

Tensor operator*(double, const Tensor &);
