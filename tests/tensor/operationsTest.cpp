#include <gtest/gtest.h>

#include "ops.h"
#include "tensor.h"

TEST(MatrixTest, RankMismatchTest) {
  Tensor scalar(24);
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  EXPECT_THROW(
      {
        try {
          matrix + scalar;
        } catch (const std::invalid_argument &e) {
          EXPECT_STREQ(e.what(), "Incompatible ranks: 2 and 0");
          throw;
        }
      },
      std::invalid_argument);
}

TEST(MatrixTest, MatrixDotVectorTest) {
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  Tensor vector1({2, 1}, {5, 6});
  Tensor vector2 = dot(matrix, vector1);
  EXPECT_EQ((vector2[{0, 0}]), 17);
  EXPECT_EQ((vector2[{1, 0}]), 39);
}

TEST(MatrixTest, MatrixDotMatrixTest) {
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

TEST(MatrixTest, PermuteTest) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2 = permute(matrix1, {1, 0});
  EXPECT_EQ((matrix2[{0, 0}]), 1);
  EXPECT_EQ((matrix2[{0, 1}]), 3);
  EXPECT_EQ((matrix2[{1, 0}]), 2);
  EXPECT_EQ((matrix2[{1, 1}]), 4);
}
