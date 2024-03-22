#include "array.h"

namespace gradstudent {

void checkEqualSize(const array_t &lhs, const array_t &rhs) {
  if (lhs != rhs) {
    std::stringstream ss;
    ss << "Expected equal size arrays, got sizes " << lhs.size() << " and "
       << rhs.size();
    throw std::invalid_argument(ss.str());
  }
}

array_t operator+(const array_t &lhs, const array_t &rhs) {
  checkEqualSize(lhs, rhs);
  array_t result(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    result[i] = lhs[i] + rhs[i];
  }
  return result;
}

array_t operator-(const array_t &lhs, const array_t &rhs) {
  checkEqualSize(lhs, rhs);
  array_t result(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    result[i] = lhs[i] - rhs[i];
  }
  return result;
}

} // namespace gradstudent
