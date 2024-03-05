#pragma once

#include <initializer_list>
#include <memory>

#include "array.h"

class Tensor {

private:
  size_t size;
  size_t ndims;
  Array shape;
  Array strides;
  std::shared_ptr<double[]> data;

  Tensor(size_t, size_t, const Array &, const Array &);

  size_t toIndex(const Array &) const;
  void checkCompatibleShape(const Tensor &) const;

public:
  Tensor(const Array &);
  Tensor(std::initializer_list<size_t>);
  ~Tensor();

  double operator[](size_t) const;
  double &operator[](size_t);
  Tensor operator+(const Tensor &) const;
  Tensor operator-() const;
  Tensor operator-(const Tensor &) const;

  Tensor dot(const Tensor &) const;

  friend Tensor operator*(double, const Tensor&);
  friend Tensor scalarTensor(double);
};

Tensor operator*(double, const Tensor&);
Tensor scalarTensor(double);
