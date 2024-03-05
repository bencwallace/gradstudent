#include <cassert>
#include <cstring>
#include <sstream>

#include "tensor.h"
#include "array.h"

int main() {
  Array array1({10});
  std::stringstream ss;
  ss << array1;
  assert(ss.str() == "(10,)");
  ss.str(std::string());

  Array array2({1, 2, 3});
  ss << array2;
  assert(ss.str() == "(1, 2, 3)");

  Tensor scalar1 = scalarTensor(42);
  assert(scalar1[{0}] == 42);

  scalar1[{0}] = 24;
  assert(scalar1[{0}] == 24);

  Tensor sum = scalar1 + scalar1;
  assert(sum[{0}] == 48);

  Tensor multiple = 5 * scalar1;
  assert (multiple[{0}] == 120);

  Tensor diff = scalar1 - scalar1;
  assert(diff[{0}] == 0);

  Tensor matrix1({2, 2});
  try {
    matrix1 + scalar1;
  } catch (const std::exception &e) {
    assert(!std::strcmp(e.what(), "Incompatible ranks: 2 and 1"));
  }

  matrix1[{0, 0}] = 1; matrix1[{0, 1}] = 2; matrix1[{1, 0}] = 3; matrix1[{1, 1}] = 4;
  assert((matrix1[{0, 0}] == 1) && (matrix1[{0, 1}] == 2) && (matrix1[{1, 0}] == 3) && (matrix1[{1, 1}] == 4));

  // TODO: simplify tensor initialization -- not this requires use of multiIndex
  Tensor vector1({2, 1});
  vector1[{0, 0}] = 5; vector1[{1, 0}] = 6;
  Tensor vector2 = matrix1.dot(vector1);
  assert((vector2[{0, 0}] == 17));
  assert((vector2[{1, 0}] == 39));

  Tensor matrix2({2, 3});
  matrix2[{0, 0}] = 6; matrix2[{0, 1}] = 5; matrix2[{0, 2}] = 4;
  matrix2[{1, 0}] = 3; matrix2[{1, 1}] = 2; matrix2[{1, 2}] = 1;
  Tensor matrix3 = matrix1.dot(matrix2);
  assert((matrix3[{0, 0}] == 12));
  assert((matrix3[{0, 1}] == 9));
  assert((matrix3[{0, 2}] == 6));
  assert((matrix3[{1, 0}] == 30));
  assert((matrix3[{1, 1}] == 23));
  assert((matrix3[{1, 2}] == 16));

  // TODO: add test summing tensors with different strides

  std::cout << "Tests passed" << std::endl;
}
