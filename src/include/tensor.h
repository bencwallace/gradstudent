#ifndef TENSOR_H
#define TENSOR_H

#include <initializer_list>
#include <memory>

#include "utils.h"

class Tensor {

private:
  size_t size;
  size_t ndims;
  Array shape;
  Array strides;
  std::shared_ptr<double[]> data;

  Tensor(size_t, size_t, const Array &, const Array &);

  size_t toIndex(const Array &) const;

public:
  Tensor(const Array &);
  Tensor(std::initializer_list<size_t>);
  Tensor(double);
  ~Tensor();

  double operator[](size_t) const;
  double &operator[](size_t);
  Tensor operator+(const Tensor &) const;

  Tensor dot(const Tensor &) const;
};

#endif
