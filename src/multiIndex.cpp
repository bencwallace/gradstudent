#include <sstream>
#include <stdexcept>

#include "multiIndex.h"

MultiIndex::MultiIndex(const Array &shape)
    : Array(zerosArray(shape.size)), shape(shape) {}

// void Multi
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
    (*this)[i] = 0;
  }
}

MultiIndex MultiIndex::operator++() {
  if (shape.size > 0) {
    increment(shape.size - 1);
  }
  return *this;
}
