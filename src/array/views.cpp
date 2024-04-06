#include "gradstudent/array.h"

namespace gs {

array_t Array::slice(size_t start, size_t stop) const {
  array_t result = array_t(stop - start, sentinel{});
  for (size_t i = 0; i < result.size(); ++i) {
    result.data_[i] = data_[start + i];
  }
  return result;
}

array_t Array::sliceFrom(size_t start) const { return slice(start, size_); }

array_t Array::sliceTo(size_t stop) const { return slice(0, stop); }

} // namespace gs
