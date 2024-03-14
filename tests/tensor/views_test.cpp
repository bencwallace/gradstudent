#include <gtest/gtest.h>

#include "tensor.h"

TEST(SliceTest, GetSlice) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor slice = matrix1.slice(array_t{0});
  EXPECT_EQ(slice.shape(), array_t({2}));
  EXPECT_EQ(slice, Tensor({2}, {1, 2}));
  slice[0] = 0;
  EXPECT_EQ(matrix1[0], 0);
  EXPECT_EQ(matrix1[1], 2);
  EXPECT_EQ(matrix1[2], 3);
  EXPECT_EQ(matrix1[3], 4);
}

TEST(SliceTest, SetSlice) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor vector1({2}, {1}, {5, 6});
  matrix1.slice(array_t{0}) = vector1;
  EXPECT_EQ(matrix1[0], 5);
  EXPECT_EQ(matrix1[1], 6);
  EXPECT_EQ(matrix1[2], 3);
  EXPECT_EQ(matrix1[3], 4);
}
