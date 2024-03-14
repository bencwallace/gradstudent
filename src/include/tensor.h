#pragma once

#include <initializer_list>
#include <memory>

#include "multi_index.h"
#include "tensor_data.h"
#include "utils.h"

class Tensor {

private:
  const size_t offset_;
  const size_t size_;
  const array_t shape_;
  const array_t strides_;
  const std::shared_ptr<TensorData> data_;

  inline size_t toIndex(const array_t &mIdx) const {
    return sumProd(mIdx, strides_);
  }
  inline size_t toIndex(const MultiIndex &mIdx) const {
    return sumProd(mIdx, strides_);
  }

  void checkCompatibleShape(const Tensor &) const;

public:
  Tensor(const Tensor &);
  explicit Tensor(const array_t &shape, const array_t &strides, const Tensor &,
                  size_t offset = 0);

  Tensor(const array_t &shape);

  Tensor(const array_t &shape, const array_t &strides,
         std::initializer_list<double> data);
  Tensor(const array_t &shape, std::initializer_list<double> data);
  Tensor(double);

  Tensor &operator=(const Tensor &);
  double operator[](size_t) const;
  double &operator[](size_t);
  double operator[](const array_t &) const;
  double &operator[](const array_t &);
  double operator[](const MultiIndex &mIdx) const;
  double &operator[](const MultiIndex &mIdx);
  Tensor operator+(const Tensor &) const;
  Tensor operator-() const;
  Tensor operator-(const Tensor &) const;
  Tensor operator*(const Tensor &) const;
  bool operator==(const Tensor &) const;
  explicit operator double() const;

  MultiIndexIter multiIndexRange() const;

  Tensor slice(const array_t &mIdx);

  size_t size() const;
  size_t ndims() const;
  const array_t &shape() const;
  const array_t &strides() const;

  friend Tensor operator*(double, const Tensor &);
};

Tensor operator*(double, const Tensor &);
