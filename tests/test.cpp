#include <cassert>

#include "tensor.h"
#include "utils.h"

int main() {
  Array test(10);

  Tensor scalar(42);
  assert(scalar[0] == 42);

  scalar[0] = 24;
  assert(scalar[0] == 24);

  Tensor sum = scalar + scalar;
  assert(sum[0] == 48);

  Tensor tensor({4, 4, 3});
}
