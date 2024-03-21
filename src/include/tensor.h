#pragma once

#include <initializer_list>
#include <memory>

#include "multi_index.h"
#include "types.h"
#include "utils.h"

namespace gradstudent {

class Tensor {

private:
  bool ro_ = false; // read-only (for views of const tensors)
  const size_t offset_;
  const size_t size_;
  const array_t shape_;
  const array_t strides_;
  std::shared_ptr<double[]> data_; // NOLINT(cppcoreguidelines-avoid-c-arrays)

  void ensureWritable();

  void assignOther(const Tensor &);
  void assignSelf(const Tensor &);

public:
  /* CONSTRUCTORS */
  Tensor(const Tensor &);
  explicit Tensor(const array_t &shape, const array_t &strides, const Tensor &,
                  size_t offset = 0, bool ro = false);

  Tensor(const array_t &shape);
  Tensor(const array_t &shape, const array_t &strides,
         std::initializer_list<double> data);
  Tensor(const array_t &shape, std::initializer_list<double> data);
  Tensor(double);

  ~Tensor() = default;

  /* OPERATORS */

  Tensor &operator=(const Tensor &);

  inline double operator[](size_t i) const { return data_[offset_ + i]; }
  inline double &operator[](size_t i) {
    ensureWritable();
    return data_[offset_ + i];
  }
  inline double operator[](const array_t &mIdx) const {
    return data_[toIndex(mIdx)];
  }
  inline double &operator[](const array_t &mIdx) {
    ensureWritable();
    return data_[toIndex(mIdx)];
  }
  inline double operator[](const MultiIndex &mIdx) const {
    return data_[toIndex(mIdx)];
  }
  inline double &operator[](const MultiIndex &mIdx) {
    ensureWritable();
    return data_[toIndex(mIdx)];
  }

  explicit operator double() const;

  /* UTILITIES */

  inline size_t toIndex(const array_t &mIdx) const {
    return sumProd(mIdx, strides_);
  }
  inline size_t toIndex(const MultiIndex &mIdx) const {
    return sumProd(mIdx, strides_);
  }

  inline MultiIndexIter multiIndexRange() const {
    return MultiIndexIter(shape_);
  }

  /* GETTERS/SETTERS */

  inline size_t size() const { return size_; }
  inline size_t ndims() const { return shape_.size(); }
  inline const array_t &shape() const { return shape_; }
  inline const array_t &strides() const { return strides_; }

  /* FRIEND OPERATORS */

  friend Tensor operator+(const Tensor &, const Tensor &);
  friend Tensor operator-(const Tensor &);
  friend Tensor operator-(const Tensor &, const Tensor &);
  friend Tensor operator*(const Tensor &, const Tensor &);
  friend bool operator==(const Tensor &, const Tensor &);
};

} // namespace gradstudent
