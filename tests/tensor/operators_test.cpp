#include <gtest/gtest.h>

#include "gradstudent/iter.h"
#include "gradstudent/tensor.h"

using namespace gs;

TEST(CastTest, Scalar) {
  Tensor scalar(24);
  EXPECT_EQ(static_cast<double>(scalar), 24);
}

TEST(CastTest, Vector) {
  Tensor vector = Tensor(24).reshape({1});
  EXPECT_EQ(static_cast<double>(vector), 24);
}

TEST(CastTest, Matrix) {
  Tensor matrix = Tensor(24).reshape({1, 1});
  EXPECT_EQ(static_cast<double>(matrix), 24);
}

TEST(CastTest, NonScalar) {
  Tensor vector = Tensor::fill({2}, 24);
  EXPECT_THROW((void)static_cast<double>(vector), std::invalid_argument);
}

TEST(SubscriptGetTest, Empty) {
  Tensor scalar(24);
  EXPECT_EQ(scalar[{}], 24);
}

TEST(SubscriptGetTest, DefaultStrides) {
  Tensor matrix = Tensor::range(1, 5).reshape({2, 2});
  EXPECT_EQ((matrix[{0, 0}]), 1);
  EXPECT_EQ((matrix[{0, 1}]), 2);
  EXPECT_EQ((matrix[{1, 0}]), 3);
  EXPECT_EQ((matrix[{1, 1}]), 4);
}

TEST(SubscriptGetTest, CustomStrides) {
  Tensor matrix = Tensor::range(1, 5).reshape({2, 2}, {1, 2});
  EXPECT_EQ((matrix[{0, 0}]), 1);
  EXPECT_EQ((matrix[{0, 1}]), 3);
  EXPECT_EQ((matrix[{1, 0}]), 2);
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
  Tensor matrix = Tensor::range(5, 9).reshape({2, 2});
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
  Tensor matrix = Tensor::range(5, 9).reshape({2, 2}, {1, 2});
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
  Tensor vector2 = Tensor::range(1, 5).reshape({4});
  vector1 = vector2;
  EXPECT_EQ(vector1.shape(), array_t({4}));
  EXPECT_EQ(vector1, vector2);
}

TEST(CopyTest, Matrix) {
  Tensor matrix1(array_t{2, 2});
  Tensor matrix2 = Tensor::range(1, 5).reshape({2, 2});
  matrix1 = matrix2;
  EXPECT_EQ(matrix1.shape(), array_t({2, 2}));
  EXPECT_EQ(matrix1, matrix2);
}

TEST(CopyTest, StridedMatrix) {
  Tensor matrix1 = Tensor::fill({2, 2}, {2, 1}, 0);
  Tensor matrix2 = Tensor::range(1, 5).reshape({2, 2}, {1, 2});
  matrix1 = matrix2;
  EXPECT_EQ((matrix1[{0, 0}]), 1);
  EXPECT_EQ((matrix1[{0, 1}]), 3);
  EXPECT_EQ((matrix1[{1, 0}]), 2);
  EXPECT_EQ((matrix1[{1, 1}]), 4);
}

TEST(AssignTest, DifferentShapes) {
  Tensor matrix1 = Tensor::range(1, 5).reshape({2, 2});
  Tensor matrix2 = Tensor::range(1, 5).reshape({4});
  EXPECT_THROW(matrix1 = matrix2, std::invalid_argument);
}

TEST(AssignTest, SameShape) {
  Tensor matrix1 = Tensor::range(1, 5).reshape({2, 2});
  Tensor matrix2 = Tensor::range(4, 0, -1).reshape({2, 2});
  matrix1 = matrix2;
  EXPECT_EQ(matrix1[0], 4);
  EXPECT_EQ(matrix1[1], 3);
  EXPECT_EQ(matrix1[2], 2);
  EXPECT_EQ(matrix1[3], 1);
  matrix1[0] = 0;
  EXPECT_EQ(matrix2[0], 4);
}

TEST(AssignTest, SameBuffer) {
  Tensor matrix1 = Tensor::range(1, 5).reshape({2, 2}, {2, 1});
  Tensor matrix2 = Tensor(matrix1.shape(), {1, 2}, matrix1);
  matrix1 = matrix2;
  EXPECT_EQ((matrix1[{0, 0}]), 1);
  EXPECT_EQ((matrix1[{0, 1}]), 3);
  EXPECT_EQ((matrix1[{1, 0}]), 2);
  EXPECT_EQ((matrix1[{1, 1}]), 4);
}
