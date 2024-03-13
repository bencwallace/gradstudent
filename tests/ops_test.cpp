#include "ops.h"
#include "tensor.h"

#include <gtest/gtest.h>

TEST(DotTest, MatrixVector) {
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  Tensor vector1({2, 1}, {5, 6});
  Tensor vector2 = dot(matrix, vector1);
  EXPECT_EQ((vector2[{0, 0}]), 17);
  EXPECT_EQ((vector2[{1, 0}]), 39);
}

TEST(DotTest, MatrixMatrix) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 3}, {6, 5, 4, 3, 2, 1});
  Tensor matrix3 = dot(matrix1, matrix2);
  EXPECT_EQ((matrix3[{0, 0}]), 12);
  EXPECT_EQ((matrix3[{0, 1}]), 9);
  EXPECT_EQ((matrix3[{0, 2}]), 6);
  EXPECT_EQ((matrix3[{1, 0}]), 30);
  EXPECT_EQ((matrix3[{1, 1}]), 23);
  EXPECT_EQ((matrix3[{1, 2}]), 16);
}

TEST(PermuteTest, Matrix) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2 = permute(matrix1, {1, 0});
  EXPECT_EQ((matrix2[{0, 0}]), 1);
  EXPECT_EQ((matrix2[{0, 1}]), 3);
  EXPECT_EQ((matrix2[{1, 0}]), 2);
  EXPECT_EQ((matrix2[{1, 1}]), 4);
}

TEST(NormTest, Matrix) {
  Tensor tensor({2, 2}, {1, 2, 3, 4});
  EXPECT_EQ(norm2(tensor), 30);
}
