#include <cassert>
#include <cstring>
#include <sstream>

#include "tensor.h"
#include "utils.h"

int main() {
  Array array1({10});
  std::stringstream ss;
  ss << array1;
  assert(ss.str() == "(10,)");
  ss.str(std::string());

  Array array2({1, 2, 3});
  ss << array2;
  assert(ss.str() == "(1, 2, 3)");

  Tensor scalar1(42);
  assert(scalar1[0] == 42);

  scalar1[0] = 24;
  assert(scalar1[0] == 24);

  Tensor sum = scalar1 + scalar1;
  assert(sum[0] == 48);

  Tensor matrix1({4, 5});
  try {
    matrix1 + scalar1;
  } catch (const std::exception &e) {
    assert(!std::strcmp(e.what(), "Incompatible ranks: 2 and 1"));
  }
}
