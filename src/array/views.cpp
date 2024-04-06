#include "array.h"

namespace gs {

array_t slice(const array_t &array, size_t start, size_t stop) {
  return array_t(array.begin() + start, array.begin() + stop);
}

array_t sliceFrom(const array_t &array, size_t start) {
  return slice(array, start, array.size());
}

array_t sliceTo(const array_t &array, size_t stop) {
  return slice(array, 0, stop);
}

} // namespace gs
