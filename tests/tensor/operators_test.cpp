#include <gtest/gtest.h>

#include "tensor.h"

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
  EXPECT_THROW((void)static_cast<double>(vector);, std::invalid_argument);
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

TEST(SumTest, RankMismatch) {
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
  Tensor matrix2({2, 2}, {1, 2}, {1, 2, 3, 4});
  Tensor diff = matrix1 - matrix2;

  EXPECT_EQ(diff.shape(), array_t({2, 2}));
  EXPECT_EQ(diff.size(), 4);

  EXPECT_EQ((diff[0]), 0);
  EXPECT_EQ((diff[1]), 0);
  EXPECT_EQ((diff[2]), 0);
  EXPECT_EQ((diff[3]), 0);
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
