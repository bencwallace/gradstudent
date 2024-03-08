#pragma once

#include <initializer_list>
#include <memory>

#include "array.h"
#include "tensor_data.h"

class Tensor {

private:
  size_t ndims;
  Array shape;
  Array strides;
  TensorData data;

  Tensor(const Array &, const Array &);
  Tensor(const Array &, const Array &, const TensorData &);

  size_t toIndex(const Array &, size_t, size_t) const;
  size_t toIndex(const Array &) const;
  Array toMultiIndex(size_t) const;
  void checkCompatibleShape(const Tensor &) const;

public:
  Tensor(const Array &);
  Tensor(const Array &, std::initializer_list<double>);
  Tensor(std::initializer_list<size_t>);
  Tensor(std::initializer_list<size_t>, std::initializer_list<double>);
  Tensor(double);

  double operator[](size_t) const;
  double &operator[](size_t);
  double operator[](const Array &) const;
  double &operator[](const Array &);
  double operator[](std::initializer_list<size_t>) const;
  double &operator[](std::initializer_list<size_t>);
  Tensor operator+(const Tensor &) const;
  Tensor operator-() const;
  Tensor operator-(const Tensor &) const;
  Tensor operator*(const Tensor &) const;

  size_t size() const;

  Tensor dot(const Tensor &) const;
  Tensor flatten() const;
  Tensor permute(std::initializer_list<size_t>);

  friend Tensor operator*(double, const Tensor &);
};

Tensor operator*(double, const Tensor &);
