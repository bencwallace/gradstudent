#include <gtest/gtest.h>

#include "tensor.h"

TEST(ScalarTest, Cast) {
  Tensor scalar(24);
  EXPECT_EQ(static_cast<double>(scalar), 24);
}

TEST(ScalarTest, EmptySubscript) {
  Tensor scalar(24);
  EXPECT_EQ(scalar[{}], 24);
}

TEST(ScalarTest, SumTest) {
  Tensor scalar(24);
  Tensor sum = scalar + scalar;
  EXPECT_EQ(static_cast<double>(sum), 48);
}

TEST(ScalarTest, ProdTest) {
  Tensor scalar(24);
  Tensor multiple = 5 * scalar;
  EXPECT_EQ(static_cast<double>(multiple), 120);
}

TEST(ScalarTest, DiffTest) {
  Tensor scalar(24);
  Tensor diff = scalar - scalar;
  EXPECT_EQ(static_cast<double>(diff), 0);
}

TEST(VectorTest, Copy) {
  Tensor vector1(Array({4}));
  Tensor vector2({4}, {1, 2, 3, 4});
  vector1 = vector2;
  EXPECT_TRUE(vector1 == vector2);
}

TEST(MatrixTest, GetItemTest) {
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  EXPECT_EQ((matrix[{0, 0}]), 1);
  EXPECT_EQ((matrix[{0, 1}]), 2);
  EXPECT_EQ((matrix[{1, 0}]), 3);
  EXPECT_EQ((matrix[{1, 1}]), 4);
}

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

TEST(MatrixTest, Sum) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 3, 2, 4});
  Tensor matrix3 = matrix2 + matrix1;
  EXPECT_EQ((matrix3[{0, 0}]), 2);
  EXPECT_EQ((matrix3[{0, 1}]), 5);
  EXPECT_EQ((matrix3[{1, 0}]), 5);
  EXPECT_EQ((matrix3[{1, 1}]), 8);
}

TEST(MatrixTest, SumStrides) {
  Tensor matrix1({2, 2}, {2, 1}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 2}, {1, 3, 2, 4});
  Tensor matrix3 = matrix2 + matrix1;
  EXPECT_EQ((matrix3[{0, 0}]), 2);
  EXPECT_EQ((matrix3[{0, 1}]), 4);
  EXPECT_EQ((matrix3[{1, 0}]), 6);
  EXPECT_EQ((matrix3[{1, 1}]), 8);
}

TEST(MatrixTest, ProdTest) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 3, 2, 4});
  Tensor matrix3 = matrix2 * matrix1;
  EXPECT_EQ((matrix3[{0, 0}]), 1);
  EXPECT_EQ((matrix3[{0, 1}]), 6);
  EXPECT_EQ((matrix3[{1, 0}]), 6);
  EXPECT_EQ((matrix3[{1, 1}]), 16);
}

TEST(MatrixTest, SliceTest) {
  const Tensor matrix1({2, 2}, {1, 2, 3, 4});
  EXPECT_EQ(matrix1.slice(Array{0}), Tensor({2}, {1, 2}));
}
