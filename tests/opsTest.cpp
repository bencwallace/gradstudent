#include "kernels.h"
#include "tensor.h"

#include <gtest/gtest.h>

TEST(NormTest, Norm2Test) {
  Tensor tensor({2, 2}, {1, 2, 3, 4});
  EXPECT_EQ(norm2(tensor), 30);
}
