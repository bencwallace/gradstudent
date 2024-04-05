#include "ops.h"
#include "tensor.h"

#include <gtest/gtest.h>

using namespace gs;

TEST(DotTest, MatrixVector) {
  Tensor matrix = Tensor::range(1, 5).reshape({2, 2});
  Tensor vector1 = Tensor::range(5, 7).reshape({2, 1});
  Tensor vector2 = dot(matrix, vector1);
  EXPECT_EQ((vector2[{0, 0}]), 17);
  EXPECT_EQ((vector2[{1, 0}]), 39);
}

TEST(DotTest, MatrixMatrix) {
  Tensor matrix1 = Tensor::range(1, 5).reshape({2, 2});
  Tensor matrix2 = Tensor::range(6, 0, -1).reshape({2, 3});
  Tensor matrix3 = dot(matrix1, matrix2);
  EXPECT_EQ((matrix3[{0, 0}]), 12);
  EXPECT_EQ((matrix3[{0, 1}]), 9);
  EXPECT_EQ((matrix3[{0, 2}]), 6);
  EXPECT_EQ((matrix3[{1, 0}]), 30);
  EXPECT_EQ((matrix3[{1, 1}]), 23);
  EXPECT_EQ((matrix3[{1, 2}]), 16);
}

TEST(DotTest, MatrixTensor) {
  Tensor matrix({2, 3});
  Tensor tensor({3, 2, 2});
  Tensor result = dot(matrix, tensor);
  EXPECT_EQ(result.shape(), (array_t{2, 2, 2}));
}

TEST(DotTest, TensorMatrix) {
  Tensor tensor({4, 5, 2, 3});
  Tensor matrix({3, 2});
  Tensor result = dot(tensor, matrix);
  EXPECT_EQ(result.shape(), (array_t{4, 5, 2, 2}));
}

TEST(NormTest, Matrix) {
  Tensor tensor = Tensor::range(1, 5).reshape({2, 2});
  EXPECT_EQ(norm2(tensor), 30);
}
