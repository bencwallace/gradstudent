#include <cstring>
#include <sstream>
#include <stdexcept>

#include "multi_index.h"
#include "utils.h"

namespace gradstudent {

/* MultiIndex */

// CONSTRUCTORS

MultiIndex::MultiIndex(const array_t &shape)
    : data_(std::make_unique<size_t[]>(shape.size())), shape_(shape) {
  std::fill_n(data_.get(), size(), 0);
}

MultiIndex::MultiIndex(const MultiIndex &other)
    : data_(std::make_unique<size_t[]>(other.size())), shape_(other.shape_) {
  std::memcpy(data_.get(), other.data_.get(), size() * sizeof(size_t));
}

// OPERATORS

bool MultiIndex::operator==(const MultiIndex &other) const {
  if (size() != other.size()) {
    return false;
  }
  for (size_t i = 0; i < size(); ++i) {
    if ((*this)[i] != other[i]) {
      return false;
    }
  }
  return true;
}

void MultiIndex::increment(size_t currDim) {
  if ((*this)[currDim] < shape_[currDim] - 1) {
    ++(*this)[currDim];
  } else if (currDim > 0) {
    (*this)[currDim] = 0;
    return increment(currDim - 1);
  } else {
    setToEnd();
  }
}

MultiIndex MultiIndex::operator++() {
  if (size() > 0) {
    increment(size() - 1);
  }
  return *this;
}

MultiIndex &MultiIndex::operator=(const MultiIndex &other) {
  if (shape_ != other.shape_) {
    std::stringstream ss;
    ss << "Expected multi-indices of equal shape, got shapes " << shape_
       << "and " << other.shape_;
  }
  std::memcpy(data_.get(), other.data_.get(), size() * sizeof(size_t));
  return *this;
}

// GETTERS/SETTERS

void MultiIndex::setToEnd() {
  if (size() > 0) {
    data_[0] = shape_[0];
    std::fill_n(&data_[1], size() - 1, 0);
  }
  isEnd_ = true;
}

MultiIndex::operator array_t() const {
  array_t result(size());
  std::memcpy(&result[0], data_.get(), size());
  return result;
}

/* MultiIndexIter */

// CONSTRUCTORS

MultiIndexIter MultiIndexIter::begin() { return *this; }

MultiIndexIter MultiIndexIter::end() { return MultiIndexIter(shape(), true); }

MultiIndexIter::MultiIndexIter(const MultiIndexIter &other)
    : curr(new value_type(*other.curr)) {}

MultiIndexIter::MultiIndexIter(const array_t &shape, bool end)
    : curr(new MultiIndex(shape)) {
  if (end) {
    curr->setToEnd();
  }
}

MultiIndexIter::~MultiIndexIter() {
  if (curr) {
    delete curr;
  }
}

// OPERATORS

MultiIndexIter &MultiIndexIter::operator=(const MultiIndexIter &other) {
  *curr = *other.curr;
  return *this;
}

MultiIndexIter::reference MultiIndexIter::operator*() const {
  if (curr) {
    return *curr;
  }
  throw std::out_of_range("Iteration complete.");
}

MultiIndexIter &MultiIndexIter::operator++() {
  if (curr->size() > 0) {
    ++(*curr);
  } else {
    curr->setToEnd();
  }
  return *this;
}

MultiIndexIter MultiIndexIter::operator++(int) {
  MultiIndexIter temp = *this;
  ++(*this);
  return temp;
}

bool operator==(const MultiIndexIter &a, const MultiIndexIter &b) {
  if (a.curr->size() == 0 || b.curr->size() == 0) {
    if (a.curr->size() > 0 || b.curr->size() > 0) {
      return false;
    }
    return a.curr->isEnd() == b.curr->isEnd();
  }
  return *a.curr == *b.curr;
};

} // namespace gradstudent
