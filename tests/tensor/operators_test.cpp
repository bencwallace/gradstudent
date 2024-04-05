#include <gtest/gtest.h>

#include "tensor.h"

using namespace gs;

TEST(CastTest, Scalar) {
  Tensor scalar(24);
  EXPECT_EQ(static_cast<double>(scalar), 24);
}

TEST(CastTest, Vector) {
  Tensor vector({1}, {24});
  EXPECT_EQ(static_cast<double>(vector), 24);
}

TEST(CastTest, Matrix) {
  Tensor matrix({1, 1}, {24});
  EXPECT_EQ(static_cast<double>(matrix), 24);
}

TEST(CastTest, NonScalar) {
  Tensor vector({2}, {24, 24});
  EXPECT_THROW((void)static_cast<double>(vector), std::invalid_argument);
}

TEST(SubscriptGetTest, Empty) {
  Tensor scalar(24);
  EXPECT_EQ(scalar[{}], 24);
}

TEST(SubscriptGetTest, DefaultStrides) {
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  EXPECT_EQ((matrix[{0, 0}]), 1);
  EXPECT_EQ((matrix[{0, 1}]), 2);
  EXPECT_EQ((matrix[{1, 0}]), 3);
  EXPECT_EQ((matrix[{1, 1}]), 4);
}

TEST(SubscriptGetTest, CustomStrides) {
  Tensor matrix({2, 2}, {1, 2}, {1, 3, 2, 4});
  EXPECT_EQ((matrix[{0, 0}]), 1);
  EXPECT_EQ((matrix[{0, 1}]), 2);
  EXPECT_EQ((matrix[{1, 0}]), 3);
  EXPECT_EQ((matrix[{1, 1}]), 4);
}

TEST(SubscriptGetTest, Const) {
  const Tensor scalar(24);
  EXPECT_EQ(scalar[{}], 24);
}

TEST(SubscriptSetTest, Empty) {
  Tensor scalar(24);
  scalar[{}] = 42;
  EXPECT_EQ(scalar[0], 42);
}

TEST(SubscriptSetTest, DefaultStrides) {
  Tensor matrix({2, 2}, {5, 6, 7, 8});
  matrix[{0, 0}] = 1;
  matrix[{0, 1}] = 2;
  matrix[{1, 0}] = 3;
  matrix[{1, 1}] = 4;

  EXPECT_EQ((matrix[0]), 1);
  EXPECT_EQ((matrix[1]), 2);
  EXPECT_EQ((matrix[2]), 3);
  EXPECT_EQ((matrix[3]), 4);
}

TEST(SubscriptSetTest, CustomStrides) {
  Tensor matrix({2, 2}, {1, 2}, {5, 6, 7, 8});
  matrix[{0, 0}] = 1;
  matrix[{0, 1}] = 2;
  matrix[{1, 0}] = 3;
  matrix[{1, 1}] = 4;

  EXPECT_EQ((matrix[0]), 1);
  EXPECT_EQ((matrix[2]), 2);
  EXPECT_EQ((matrix[1]), 3);
  EXPECT_EQ((matrix[3]), 4);
}

TEST(CopyTest, Vector) {
  Tensor vector1(array_t({4}));
  Tensor vector2({4}, {1, 2, 3, 4});
  vector1 = vector2;
  EXPECT_EQ(vector1.shape(), array_t({4}));
  EXPECT_EQ(vector1, vector2);
}

TEST(CopyTest, Matrix) {
  Tensor matrix1({2, 2});
  Tensor matrix2({2, 2}, {1, 2, 3, 4});
  matrix1 = matrix2;
  EXPECT_EQ(matrix1.shape(), array_t({2, 2}));
  EXPECT_EQ(matrix1, matrix2);
}

TEST(CopyTest, StridedMatrix) {
  Tensor matrix1({2, 2}, {2, 1}, {});
  Tensor matrix2({2, 2}, {1, 2}, {1, 2, 3, 4});
  matrix1 = matrix2;
  EXPECT_EQ((matrix1[0]), 1);
  EXPECT_EQ((matrix1[1]), 3);
  EXPECT_EQ((matrix1[2]), 2);
  EXPECT_EQ((matrix1[3]), 4);
}

TEST(AssignTest, DifferentShapes) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({4}, {1, 2, 3, 4});
  EXPECT_THROW(matrix1 = matrix2, std::invalid_argument);
}

TEST(AssignTest, SameShape) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {4, 3, 2, 1});
  matrix1 = matrix2;
  EXPECT_EQ(matrix1[0], 4);
  EXPECT_EQ(matrix1[1], 3);
  EXPECT_EQ(matrix1[2], 2);
  EXPECT_EQ(matrix1[3], 1);
  matrix1[0] = 0;
  EXPECT_EQ(matrix2[0], 4);
}

TEST(AssignTest, SameBuffer) {
  Tensor matrix1({2, 2}, {2, 1}, {1, 2, 3, 4});

  Tensor matrix2 = Tensor(matrix1.shape(), {1, 2}, matrix1);
  // EXPECT_EQ((matrix2[{0, 0}]), 1);
  // EXPECT_EQ((matrix2[{0, 1}]), 3);
  // EXPECT_EQ((matrix2[{1, 0}]), 2);
  // EXPECT_EQ((matrix2[{1, 1}]), 4);
  matrix1 = matrix2;
  EXPECT_EQ((matrix1[{0, 0}]), 1);
  EXPECT_EQ((matrix1[{0, 1}]), 3);
  EXPECT_EQ((matrix1[{1, 0}]), 2);
  EXPECT_EQ((matrix1[{1, 1}]), 4);
}
