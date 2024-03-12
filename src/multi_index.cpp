#include <sstream>
#include <stdexcept>

#include "multi_index.h"

/* MultiIndex */

MultiIndex::MultiIndex(const Array &shape)
    : Array(zerosArray(shape.size)), shape(shape) {}

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
    reset();
  }
}

void MultiIndex::reset() {
  for (size_t i = 0; i < shape.size; ++i) {
    (*this)[i] = -1;
  }
}

MultiIndex MultiIndex::operator++() {
  if (shape.size > 0) {
    increment(shape.size - 1);
  }
  return *this;
}

/* MultiIndexRange */

MultiIndexRange::MultiIndexRange(const Array &shape)
    : shape(shape) {}

using MultiIndexIter = MultiIndexRange::MultiIndexIter;

MultiIndexIter MultiIndexRange::begin() {
  return MultiIndexIter(shape);
}

MultiIndexIter MultiIndexRange::end() {
  return MultiIndexIter(shape, true);
}

/* MultiIndexIter */

MultiIndexIter::MultiIndexIter(const Array &shape, bool end)
    : curr(new MultiIndex(shape)) {
  if (end) {
    if (shape.size > 0) {
      curr->reset();
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

MultiIndexIter::reference MultiIndexIter::operator*() const {
  if (curr) {
    return *curr;
  }
  throw std::out_of_range("Iteration complete.");
}

MultiIndexIter::pointer MultiIndexIter::operator->() { return curr; }

MultiIndexIter& MultiIndexIter::operator++() {
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

bool operator==(const MultiIndexIter& a, const MultiIndexIter& b) {
  // if (a.curr == nullptr || b.curr == nullptr) {
  //   return false;
  // }
  if (a.curr == nullptr || b.curr == nullptr) {
    return a.curr == b.curr;
  }
  return *a.curr == *b.curr;
};

bool operator!=(const MultiIndexIter& a, const MultiIndexIter& b) {
  if (a.curr == nullptr || b.curr == nullptr) {
    return a.curr != b.curr;
  }
  return *a.curr != *b.curr;
};
