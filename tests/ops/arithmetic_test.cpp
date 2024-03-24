#include <gtest/gtest.h>

#include "tensor.h"

using namespace gradstudent;

TEST(SumTest, Scalar) {
  Tensor scalar(24);
  Tensor sum = scalar + scalar;
  EXPECT_EQ(sum.shape(), array_t{});
  EXPECT_EQ(sum.size(), 1);
  EXPECT_EQ(sum[0], 48);
}

TEST(SumTest, Matrix) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 3, 2, 4});
  Tensor matrix3 = matrix2 + matrix1;

  EXPECT_EQ(matrix3.shape(), array_t({2, 2}));
  EXPECT_EQ(matrix3.size(), 4);

  EXPECT_EQ((matrix3[0]), 2);
  EXPECT_EQ((matrix3[1]), 5);
  EXPECT_EQ((matrix3[2]), 5);
  EXPECT_EQ((matrix3[3]), 8);
}

TEST(SumTest, StridedMatrix) {
  Tensor matrix1({2, 2}, {2, 1}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 2}, {1, 3, 2, 4});
  Tensor matrix3 = matrix2 + matrix1;

  EXPECT_EQ(matrix3.shape(), array_t({2, 2}));
  EXPECT_EQ(matrix3.size(), 4);

  EXPECT_EQ((matrix3[0]), 2);
  EXPECT_EQ((matrix3[1]), 4);
  EXPECT_EQ((matrix3[2]), 6);
  EXPECT_EQ((matrix3[3]), 8);
}

TEST(SumTest, Broadcast) {
  Tensor vector1({1, 3}, {0, 1, 2});
  Tensor vector2({3, 1}, {0, 1, 2});
  Tensor vector3 = vector1 + vector2;
  EXPECT_EQ(vector3.shape(), (array_t{3, 3}));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ((vector3[{i, j}]), i + j) << "i, j == " << i << ", " << j;
    }
  }
}

TEST(ScalarProdTest, Scalar) {
  Tensor scalar(24);
  Tensor multiple = 5 * scalar;

  EXPECT_EQ(multiple.shape(), array_t{});
  EXPECT_EQ(multiple.size(), 1);
  EXPECT_EQ(multiple[0], 120);
}

TEST(ScalarProdTest, Matrix) {
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  Tensor multiple = 5 * matrix;

  EXPECT_EQ(multiple.shape(), array_t({2, 2}));
  EXPECT_EQ(multiple.size(), 4);

  EXPECT_EQ((multiple[0]), 5);
  EXPECT_EQ((multiple[1]), 10);
  EXPECT_EQ((multiple[2]), 15);
  EXPECT_EQ((multiple[3]), 20);
}

TEST(ScalarProdTest, StridedMatrix) {
  Tensor matrix({2, 2}, {1, 2}, {1, 3, 2, 4});
  Tensor multiple = 5 * matrix;

  EXPECT_EQ(multiple.shape(), array_t({2, 2}));
  EXPECT_EQ(multiple.size(), 4);

  EXPECT_EQ((multiple[0]), 5);
  EXPECT_EQ((multiple[1]), 10);
  EXPECT_EQ((multiple[2]), 15);
  EXPECT_EQ((multiple[3]), 20);
}

TEST(DiffTest, Scalar) {
  Tensor scalar(24);
  Tensor diff = scalar - scalar;
  EXPECT_EQ(diff.shape(), array_t{});
  EXPECT_EQ(diff.size(), 1);
  EXPECT_EQ(diff[0], 0);
}

TEST(DiffTest, Matrix) {
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  Tensor diff = matrix - matrix;

  EXPECT_EQ(diff.shape(), array_t({2, 2}));
  EXPECT_EQ(diff.size(), 4);

  EXPECT_EQ((diff[0]), 0);
  EXPECT_EQ((diff[1]), 0);
  EXPECT_EQ((diff[2]), 0);
  EXPECT_EQ((diff[3]), 0);
}

TEST(DiffTest, StridedMatrix) {
  Tensor matrix1({2, 2}, {2, 1}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 2}, {1, 3, 2, 4});

  Tensor diff = matrix1 - matrix2;
  EXPECT_EQ(diff.shape(), array_t({2, 2}));
  EXPECT_EQ(diff.size(), 4);

  ASSERT_EQ(matrix1, matrix2);
  EXPECT_EQ((diff[0]), 0);
  EXPECT_EQ((diff[1]), 0);
  EXPECT_EQ((diff[2]), 0);
  EXPECT_EQ((diff[3]), 0);
}

TEST(DiffTest, Broadcast) {
  Tensor vector1({1, 3}, {0, 1, 2});
  Tensor vector2({3, 1}, {0, 1, 2});
  Tensor vector3 = vector1 - vector2;
  EXPECT_EQ(vector3.shape(), (array_t{3, 3}));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ((vector3[{(size_t)i, (size_t)j}]), j - i)
          << "i, j == " << i << ", " << j;
    }
  }
}

TEST(ProdTest, Matrix) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 3, 2, 4});
  Tensor matrix3 = matrix2 * matrix1;

  EXPECT_EQ(matrix3.shape(), array_t({2, 2}));

  EXPECT_EQ((matrix3[0]), 1);
  EXPECT_EQ((matrix3[1]), 6);
  EXPECT_EQ((matrix3[2]), 6);
  EXPECT_EQ((matrix3[3]), 16);
}

TEST(ProdTest, StridedMatrix) {
  Tensor matrix1({2, 2}, {2, 1}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 2}, {1, 3, 2, 4});
  Tensor matrix3 = matrix2 * matrix1;
  EXPECT_EQ((matrix3[0]), 1);
  EXPECT_EQ((matrix3[1]), 4);
  EXPECT_EQ((matrix3[2]), 9);
  EXPECT_EQ((matrix3[3]), 16);
}

TEST(ProdTest, Broadcast) {
  Tensor vector1({1, 3}, {0, 1, 2});
  Tensor vector2({3, 1}, {0, 1, 2});
  Tensor vector3 = vector1 * vector2;
  EXPECT_EQ(vector3.shape(), (array_t{3, 3}));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ((vector3[{i, j}]), i * j) << "i, j == " << i << ", " << j;
    }
  }
}

TEST(IsEqualTest, EqualScalar) {
  Tensor scalar1(13);
  Tensor scalar2(13);
  EXPECT_TRUE(scalar1 == scalar2);
}

TEST(IsEqualTest, UnequalScalar) {
  Tensor scalar1(13);
  Tensor scalar2(31);
  EXPECT_FALSE(scalar1 == scalar2);
}

TEST(IsEqualTest, EqualMatrix) {
  Tensor matrix1({2, 2}, {2, 1}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 2}, {1, 3, 2, 4});
  EXPECT_TRUE(matrix1 == matrix2);
}
