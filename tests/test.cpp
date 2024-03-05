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

  Tensor multiple = 5 * scalar1;
  assert (multiple[0] == 120);

  Tensor diff = scalar1 - scalar1;
  assert(diff[0] == 0);

  Tensor matrix1({2, 2});
  try {
    matrix1 + scalar1;
  } catch (const std::exception &e) {
    assert(!std::strcmp(e.what(), "Incompatible ranks: 2 and 1"));
  }

  matrix1[0] = 1; matrix1[1] = 2; matrix1[2] = 3; matrix1[3] = 4;
  assert(matrix1[0] == 1 && matrix1[1] == 2 && matrix1[2] == 3 && matrix1[3] == 4);

  // TODO: simplify tensor initialization
  Tensor matrix2({2, 1});
  matrix2[0] = 5; matrix2[1] = 6;
  Tensor matrix3 = matrix1.dot(matrix2);
  assert(matrix3[0] == 17);
  assert(matrix3[1] == 39);

  std::cout << "Tests passed" << std::endl;
}
