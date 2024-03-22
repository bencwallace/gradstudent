#include <cstring>
#include <sstream>
#include <stdexcept>

#include "multi_index.h"
#include "utils.h"

namespace gradstudent {

/* CONSTRUCTORS */

MultiIndexIter MultiIndexIter::begin() { return *this; }

MultiIndexIter MultiIndexIter::end() { return MultiIndexIter(shape(), true); }

MultiIndexIter::MultiIndexIter(const MultiIndexIter &other)
    : curr_(new value_type(*other.curr_)), shape_(other.shape_),
      isEnd_(other.isEnd_) {}

void MultiIndexIter::setToEnd() {
  if (curr_->size() > 0) {
    (*curr_)[0] = shape_[0];
    std::fill_n(&(*curr_)[1], curr_->size() - 1, 0);
  }
  isEnd_ = true;
}

MultiIndexIter::MultiIndexIter(const array_t &shape, bool end)
    : curr_(new value_type(shape.size(), 0)), shape_(shape), isEnd_(end) {
  if (end) {
    setToEnd();
  }
}

MultiIndexIter::~MultiIndexIter() {
  if (curr_) {
    delete curr_;
  }
}

/* OPERATORS */

MultiIndexIter &MultiIndexIter::operator=(const MultiIndexIter &other) {
  if (shape_ != other.shape_) {
    std::stringstream ss;
    ss << "Expected multi-indices of equal shape, got shapes " << shape_
       << "and " << other.shape_;
    throw std::invalid_argument(ss.str());
  }
  *curr_ = *other.curr_;
  isEnd_ = other.isEnd_;
  return *this;
}

MultiIndexIter::reference MultiIndexIter::operator*() const {
  if (curr_) {
    return *curr_;
  }
  throw std::out_of_range("Iteration complete.");
}

void MultiIndexIter::increment(size_t currDim) {
  if ((*curr_)[currDim] < shape_[currDim] - 1) {
    ++(*curr_)[currDim];
  } else if (currDim > 0) {
    (*curr_)[currDim] = 0;
    return increment(currDim - 1);
  } else {
    setToEnd();
  }
}

void MultiIndexIter::increment() {
  if (shape_.size() > 0) {
    increment(shape_.size() - 1);
  }
}

MultiIndexIter &MultiIndexIter::operator++() {
  if (curr_->size() > 0) {
    increment();
  } else {
    isEnd_ = true;
  }
  return *this;
}

MultiIndexIter MultiIndexIter::operator++(int) {
  MultiIndexIter temp = *this;
  ++(*this);
  return temp;
}

bool operator==(const MultiIndexIter &a, const MultiIndexIter &b) {
  if (a.curr_->size() == 0 || b.curr_->size() == 0) {
    if (a.curr_->size() > 0 || b.curr_->size() > 0) {
      return false;
    }
    return a.isEnd_ == b.isEnd_;
  }
  return *a.curr_ == *b.curr_;
};

} // namespace gradstudent
