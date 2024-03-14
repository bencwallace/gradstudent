#include <sstream>
#include <stdexcept>

#include "multi_index.h"
#include "utils.h"

/* MultiIndex */

MultiIndex::MultiIndex(const Array &shape, const Array &strides, size_t offset)
    : Array(zerosArray(shape.size)), shape(shape), strides(strides),
      offset(offset) {}

MultiIndex::MultiIndex(const MultiIndex &other)
    : Array(other), shape(other.shape), strides(other.strides),
      offset(other.offset) {}

bool MultiIndex::operator==(const MultiIndex &other) const {
  if (size != other.size) {
    return false;
  }
  for (size_t i = 0; i < shape.size; ++i) {
    if ((*this)[i] != other[i]) {
      return false;
    }
  }
  return true;
}

bool MultiIndex::operator!=(const MultiIndex &other) const {
  return !((*this) == other);
}

void MultiIndex::increment(size_t currDim) {
  if ((*this)[currDim] < shape[currDim] - 1) {
    ++(*this)[currDim];
  } else if (currDim > 0) {
    (*this)[currDim] = 0;
    return increment(currDim - 1);
  } else {
    setToEnd();
  }
}

MultiIndex &MultiIndex::operator=(const MultiIndex &other) {
  if (shape != other.shape) {
    std::stringstream ss;
    ss << "Expected multi-indices of equal shape, got shapes " << shape
       << "and " << other.shape;
  }
  Array::operator=(other);
  return *this;
}

void MultiIndex::setToEnd() {
  for (size_t i = 0; i < shape.size; ++i) {
    (*this)[i] = -1;
  }
}

size_t MultiIndex::toIndex(size_t start, size_t end) const {
  if (start < 0) {
    std::stringstream ss;
    ss << "Multi-index start point must be non-negative, got " << start;
    throw std::invalid_argument(ss.str());
  }
  if (end > size) {
    std::stringstream ss;
    ss << "Invalid end point " << end << " for multi-index of size " << size;
    throw std::invalid_argument(ss.str());
  }

  return offset + sumProd((*this), strides, start, end);
}

size_t MultiIndex::toIndex() const { return toIndex(0, size); }

MultiIndex MultiIndex::operator++() {
  if (shape.size > 0) {
    increment(shape.size - 1);
  }
  return *this;
}

/* MultiIndexIter */

MultiIndexIter MultiIndexIter::begin() { return *this; }

MultiIndexIter MultiIndexIter::end() {
  return MultiIndexIter(shape, strides, offset, true);
}

MultiIndexIter::MultiIndexIter(const MultiIndexIter &other)
    : shape(other.shape), strides(other.strides), offset(other.offset),
      curr(new value_type(*other.curr)) {}

MultiIndexIter::MultiIndexIter(const Array &shape, const Array &strides,
                               size_t offset, bool end)
    : shape(shape), strides(strides), offset(offset),
      curr(new MultiIndex(shape, strides, offset)) {
  if (end) {
    if (shape.size > 0) {
      curr->setToEnd();
    } else {
      delete curr;
      curr = nullptr;
    }
  }
}

MultiIndexIter::~MultiIndexIter() {
  if (curr) {
    delete curr;
  }
}

MultiIndexIter &MultiIndexIter::operator=(const MultiIndexIter &other) {
  if (other.curr == nullptr && curr != nullptr) {
    delete curr;
    curr = nullptr;
  } else if (other.curr != nullptr) {
    if (curr == nullptr) {
      curr = new MultiIndex(*other.curr);
    } else {
      *curr = *other.curr;
    }
  }
  return *this;
}

MultiIndexIter::reference MultiIndexIter::operator*() const {
  if (curr) {
    return *curr;
  }
  throw std::out_of_range("Iteration complete.");
}

MultiIndexIter::pointer MultiIndexIter::operator->() { return curr; }

MultiIndexIter &MultiIndexIter::operator++() {
  if (curr->size > 0) {
    ++(*curr);
  } else {
    delete curr;
    curr = nullptr;
  }
  return *this;
}

MultiIndexIter MultiIndexIter::operator++(int) {
  MultiIndexIter temp = *this;
  ++(*this);
  return temp;
}

bool operator==(const MultiIndexIter &a, const MultiIndexIter &b) {
  if (a.curr == nullptr || b.curr == nullptr) {
    return a.curr == b.curr;
  }
  return *a.curr == *b.curr;
};

bool operator!=(const MultiIndexIter &a, const MultiIndexIter &b) {
  if (a.curr == nullptr || b.curr == nullptr) {
    return a.curr != b.curr;
  }
  return *a.curr != *b.curr;
};
