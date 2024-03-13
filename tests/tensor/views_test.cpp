#include <gtest/gtest.h>

#include "tensor.h"


TEST(SliceTest, Matrix) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor slice = matrix1.slice(Array{0});
  EXPECT_EQ(slice.shape(), Array({2}));
  EXPECT_EQ(slice, Tensor({2}, {1, 2}));
  slice[0] = 0;
  EXPECT_EQ(matrix1[0], 0);
}
