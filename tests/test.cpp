#include <gtest/gtest.h>

#include "tensor.h"

TEST(ArrayTest, PrintScalarTest) {
  Array array1({10});
  std::stringstream ss;
  ss << array1;
  EXPECT_EQ(ss.str(), "(10,)");
}

TEST(ArrayTest, PrintTest) {
  Array array2({1, 2, 3});
  std::stringstream ss;
  ss << array2;
  EXPECT_EQ(ss.str(), "(1, 2, 3)");
}

TEST(ScalarTest, ScalarTest) {
  Tensor scalar = scalarTensor(42);
  EXPECT_EQ(scalar[{0}], 42);
}

TEST(ScalarTest, SetItemTest) {
  Tensor scalar = scalarTensor(42);
  scalar[{0}] = 24;
  EXPECT_EQ(scalar[{0}], 24);
}

TEST(ScalarTest, SumTest) {
  Tensor scalar = scalarTensor(24);
  Tensor sum = scalar + scalar;
  EXPECT_EQ(sum[{0}], 48);
}

TEST(ScalarTest, ProdTest) {
  Tensor scalar = scalarTensor(24);
  Tensor multiple = 5 * scalar;
  EXPECT_EQ(multiple[{0}], 120);
}

TEST(ScalarTest, DiffTest) {
  Tensor scalar = scalarTensor(24);
  Tensor diff = scalar - scalar;
  EXPECT_EQ(diff[{0}], 0);
}

TEST(MatrixTest, GetItemTest) {
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  EXPECT_EQ((matrix[{0, 0}]), 1);
  EXPECT_EQ((matrix[{0, 1}]), 2);
  EXPECT_EQ((matrix[{1, 0}]), 3);
  EXPECT_EQ((matrix[{1, 1}]), 4);
}

TEST(MatrixTest, RankMismatchTest) {
  Tensor scalar = scalarTensor(24);
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  EXPECT_THROW({
    try {
      matrix + scalar;
    } catch (const std::invalid_argument &e) {
      EXPECT_STREQ(e.what(), "Incompatible ranks: 2 and 1");
      throw;
    }
  }, std::invalid_argument);
}

TEST(MatrixTest, MatrixDotVectorTest) {
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  Tensor vector1({2, 1}, {5, 6});
  Tensor vector2 = matrix.dot(vector1);
  EXPECT_EQ((vector2[{0, 0}]), 17);
  EXPECT_EQ((vector2[{1, 0}]), 39);
}

TEST(MatrixTest, MatrixDotMatrixTest) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 3}, {6, 5, 4, 3, 2, 1});
  Tensor matrix3 = matrix1.dot(matrix2);
  EXPECT_EQ((matrix3[{0, 0}]), 12);
  EXPECT_EQ((matrix3[{0, 1}]), 9);
  EXPECT_EQ((matrix3[{0, 2}]), 6);
  EXPECT_EQ((matrix3[{1, 0}]), 30);
  EXPECT_EQ((matrix3[{1, 1}]), 23);
  EXPECT_EQ((matrix3[{1, 2}]), 16);
}

TEST(MatrixTest, PermuteTest) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2 = matrix1.permute({1, 0});
  EXPECT_EQ((matrix2[{0, 0}]), 1);
  EXPECT_EQ((matrix2[{0, 1}]), 3);
  EXPECT_EQ((matrix2[{1, 0}]), 2);
  EXPECT_EQ((matrix2[{1, 1}]), 4);
}

TEST(MatrixTest, SumTest) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 3, 2, 4});
  Tensor matrix3 = matrix2 + matrix1;
  EXPECT_EQ((matrix3[{0, 0}]), 2);
  EXPECT_EQ((matrix3[{0, 1}]), 5);
  EXPECT_EQ((matrix3[{1, 0}]), 5);
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
