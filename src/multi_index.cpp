#include <cstring>
#include <sstream>
#include <stdexcept>

#include "multi_index.h"
#include "utils.h"

/* MultiIndex */

MultiIndex::MultiIndex(const array_t &shape)
    : data_(std::make_unique<size_t[]>(shape.size())), shape_(shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    data_[i] = 0;
  }
}

MultiIndex::MultiIndex(const MultiIndex &other)
    : data_(std::make_unique<size_t[]>(other.size())), shape_(other.shape_) {
  for (size_t i = 0; i < size(); ++i) {
    data_[i] = other.data_[i];
  }
}

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

MultiIndex &MultiIndex::operator=(const MultiIndex &other) {
  if (shape_ != other.shape_) {
    std::stringstream ss;
    ss << "Expected multi-indices of equal shape, got shapes " << shape_
       << "and " << other.shape_;
  }
  for (size_t i = 0; i < size(); ++i) {
    data_[i] = other.data_[i];
  }
  return *this;
}

void MultiIndex::setToEnd() {
  for (size_t i = 0; i < size(); ++i) {
    (*this)[i] = -1;
  }
  isEnd_ = true;
}

MultiIndex::operator array_t() const {
  array_t result(size());
  std::memcpy(&result[0], data_.get(), size());
  return result;
}

MultiIndex MultiIndex::operator++() {
  if (size() > 0) {
    increment(size() - 1);
  }
  return *this;
}

/* MultiIndexIter */

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
