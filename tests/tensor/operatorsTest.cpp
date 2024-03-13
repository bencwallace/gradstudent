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
  EXPECT_THROW((void) static_cast<double>(vector);, std::invalid_argument);
}

TEST(SubscriptTest, Empty) {
  Tensor scalar(24);
  EXPECT_EQ(scalar[{}], 24);
}

TEST(SubscriptTest, NonEmpty) {
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  EXPECT_EQ((matrix[{0, 0}]), 1);
  EXPECT_EQ((matrix[{0, 1}]), 2);
  EXPECT_EQ((matrix[{1, 0}]), 3);
  EXPECT_EQ((matrix[{1, 1}]), 4);
}

TEST(SumTest, Scalar) {
  Tensor scalar(24);
  Tensor sum = scalar + scalar;
  EXPECT_EQ(sum.ndims(), 0);
  EXPECT_EQ(sum.shape(), Array{});
  EXPECT_EQ(sum.size(), 1);
  EXPECT_EQ(sum[0], 48);
}

TEST(SumTest, Matrix) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 3, 2, 4});
  Tensor matrix3 = matrix2 + matrix1;
  EXPECT_EQ((matrix3[{0, 0}]), 2);
  EXPECT_EQ((matrix3[{0, 1}]), 5);
  EXPECT_EQ((matrix3[{1, 0}]), 5);
  EXPECT_EQ((matrix3[{1, 1}]), 8);
}

TEST(SumTest, StridedMatrix) {
  Tensor matrix1({2, 2}, {2, 1}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 2}, {1, 3, 2, 4});
  Tensor matrix3 = matrix2 + matrix1;
  EXPECT_EQ((matrix3[{0, 0}]), 2);
  EXPECT_EQ((matrix3[{0, 1}]), 4);
  EXPECT_EQ((matrix3[{1, 0}]), 6);
  EXPECT_EQ((matrix3[{1, 1}]), 8);
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
  EXPECT_EQ(multiple.ndims(), 0);
  EXPECT_EQ(multiple.shape(), Array{});
  EXPECT_EQ(multiple.size(), 1);
  EXPECT_EQ(multiple[0], 120);
}

TEST(DiffTest, Scalar) {
  Tensor scalar(24);
  Tensor diff = scalar - scalar;
  EXPECT_EQ(diff.ndims(), 0);
  EXPECT_EQ(diff.shape(), Array{});
  EXPECT_EQ(diff.size(), 1);
  EXPECT_EQ(diff[0], 0);
}

TEST(CopyTest, Vector) {
  Tensor vector1(Array({4}));
  Tensor vector2({4}, {1, 2, 3, 4});
  vector1 = vector2;
  EXPECT_EQ(vector1, vector2);
}

TEST(ProdTest, Matrix) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {1, 3, 2, 4});
  Tensor matrix3 = matrix2 * matrix1;
  EXPECT_EQ((matrix3[{0, 0}]), 1);
  EXPECT_EQ((matrix3[{0, 1}]), 6);
  EXPECT_EQ((matrix3[{1, 0}]), 6);
  EXPECT_EQ((matrix3[{1, 1}]), 16);
}

TEST(SliceTest, Matrix) {
  const Tensor matrix1({2, 2}, {1, 2, 3, 4});
  EXPECT_EQ(matrix1.slice(Array{0}), Tensor({2}, {1, 2}));
}
