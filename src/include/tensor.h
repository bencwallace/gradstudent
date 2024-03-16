#pragma once

#include <initializer_list>
#include <memory>

#include "multi_index.h"
#include "utils.h"

class Tensor {

private:
  const size_t offset_;
  const size_t size_;
  const array_t shape_;
  const array_t strides_;
  const std::shared_ptr<double[]> data_;

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

  inline double operator[](size_t i) const { return data_[offset_ + i]; }
  inline double &operator[](size_t i) { return data_[offset_ + i]; }
  inline double operator[](const array_t &mIdx) const {
    return (*this)[toIndex(mIdx)];
  }
  inline double &operator[](const array_t &mIdx) {
    return (*this)[toIndex(mIdx)];
  }
  inline double operator[](const MultiIndex &mIdx) const {
    return (*this)[toIndex(mIdx)];
  }
  inline double &operator[](const MultiIndex &mIdx) {
    return (*this)[toIndex(mIdx)];
  }

  Tensor operator+(const Tensor &) const;
  Tensor operator-() const;
  Tensor operator-(const Tensor &) const;
  Tensor operator*(const Tensor &) const;
  bool operator==(const Tensor &) const;
  explicit operator double() const;

  inline MultiIndexIter multiIndexRange() const {
    return MultiIndexIter(shape_);
  }

  Tensor slice(const array_t &mIdx);

  inline size_t size() const { return size_; }
  inline size_t ndims() const { return shape_.size(); }
  inline const array_t &shape() const { return shape_; }
  inline const array_t &strides() const { return strides_; }

  friend Tensor operator*(double, const Tensor &);
};

Tensor operator*(double, const Tensor &);
