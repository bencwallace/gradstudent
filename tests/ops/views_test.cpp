#include <gtest/gtest.h>

#include "ops.h"
#include "tensor.h"

using namespace gs;

TEST(SliceTest, GetSlice) {
  Tensor matrix1 = Tensor::range(1, 5).reshape({2, 2});
  Tensor sliced = slice(matrix1, array_t{0});
  EXPECT_EQ(sliced.shape(), array_t({2}));
  EXPECT_EQ(sliced, Tensor::range(1, 3).reshape({2}));
  sliced[0] = 0;
  EXPECT_EQ(matrix1[0], 0);
  EXPECT_EQ(matrix1[1], 2);
  EXPECT_EQ(matrix1[2], 3);
  EXPECT_EQ(matrix1[3], 4);
}

TEST(SliceTest, SetSlice) {
  Tensor matrix1 = Tensor::range(1, 5).reshape({2, 2});
  Tensor vector1 = Tensor::range(5, 7).reshape({2}, {1});
  slice(matrix1, array_t{0}) = vector1;
  EXPECT_EQ(matrix1[0], 5);
  EXPECT_EQ(matrix1[1], 6);
  EXPECT_EQ(matrix1[2], 3);
  EXPECT_EQ(matrix1[3], 4);
}

TEST(SliceTest, GetSliceConst) {
  const Tensor matrix1 = Tensor::range(1, 5).reshape({2, 2});
  Tensor sliced = slice(matrix1, array_t{0});
  sliced[0] = 0;
  EXPECT_EQ(matrix1[0], 1);
}

TEST(SliceTest, SetSliceConst) {
  const Tensor matrix1 = Tensor::range(1, 5).reshape({2, 2});
  Tensor vector1 = Tensor::range(5, 7).reshape({2}, {1});
  Tensor sliced = slice(matrix1, array_t{0});
  sliced = vector1;
  EXPECT_EQ(matrix1[0], 1);
  EXPECT_EQ(matrix1[1], 2);
}

TEST(PermuteTest, Subscript) {
  Tensor matrix = Tensor::range(1, 7).reshape({2, 3});
  Tensor perm = permute(matrix, {1, 0});

  EXPECT_EQ(perm.ndims(), 2);
  EXPECT_EQ(perm.shape(), array_t({3, 2}));

  EXPECT_EQ((perm[{0, 0}]), 1);
  EXPECT_EQ((perm[{1, 0}]), 2);
  EXPECT_EQ((perm[{2, 0}]), 3);
  EXPECT_EQ((perm[{0, 1}]), 4);
  EXPECT_EQ((perm[{1, 1}]), 5);
  EXPECT_EQ((perm[{2, 1}]), 6);

  perm[{0, 1}] = 10;
  EXPECT_EQ((matrix[{1, 0}]), 10);
}

TEST(PermuteTest, Assign) {
  Tensor matrix1 = Tensor::range(1, 5).reshape({2, 2});
  Tensor matrix2 = Tensor::range(5, 9).reshape({2, 2});
  permute(matrix1, {1, 0}) = matrix2;
  EXPECT_EQ((matrix1[{0, 0}]), 5);
  EXPECT_EQ((matrix1[{0, 1}]), 7);
  EXPECT_EQ((matrix1[{1, 0}]), 6);
  EXPECT_EQ((matrix1[{1, 1}]), 8);
}

TEST(PermuteTest, SubscriptConst) {
  const Tensor matrix = Tensor::range(1, 7).reshape({2, 3});
  Tensor perm = permute(matrix, {1, 0});
  perm[0] = 0;
  EXPECT_EQ(matrix[0], 1);
}

TEST(PermuteTest, AssignConst) {
  const Tensor matrix1 = Tensor::range(1, 5).reshape({2, 2});
  Tensor matrix2 = Tensor::range(5, 9).reshape({2, 2});
  Tensor perm = permute(matrix1, {1, 0});
  perm = matrix2;
  EXPECT_EQ((matrix1[{0, 0}]), 1);
  EXPECT_EQ((matrix1[{0, 1}]), 2);
  EXPECT_EQ((matrix1[{1, 0}]), 3);
  EXPECT_EQ((matrix1[{1, 1}]), 4);
}

TEST(BroadcastTest, NonConst0) {
  Tensor tensor = Tensor(6).reshape({1});
  const array_t shape{8};
  Tensor result = broadcast(tensor, shape);
  EXPECT_EQ(result.shape(), shape);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(result[array_t{i}], 6);
  }

  result[array_t{3}] = 2;
  EXPECT_EQ(tensor[0], 2);
}

TEST(BroadcastTest, NonConst1) {
  Tensor tensor = Tensor::range(1, 5).reshape({4});
  const array_t shape{{1, 4}};
  Tensor result = broadcast(tensor, shape);
  EXPECT_EQ(result.shape(), shape);
  for (size_t i = 0; i < 4; ++i) {
    auto a = result[{0, i}];
    auto b = i + 1;
    EXPECT_EQ(a, b);
    result[{0, i}] = -i;
    EXPECT_EQ(tensor[i], -i);
  }
}

TEST(BroadcastTest, NonConst2) {
  Tensor tensor = Tensor::range(1, 5).reshape({4, 1});
  const array_t shape{{4, 4}};
  Tensor result = broadcast(tensor, shape);
  EXPECT_EQ(result.shape(), shape);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      EXPECT_EQ((result[{i, j}]), i + 1);
    }
    result[{i, i}] = -i;
    EXPECT_EQ(tensor[i], -i);
  }
}

TEST(BroadcastTest, Const0) {
  const Tensor tensor = Tensor(6).reshape({1});
  const array_t shape{8};
  auto result = broadcast(tensor, shape);
  EXPECT_EQ(result.shape(), array_t{8});
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(result[array_t{i}], 6);
  }

  result[array_t{3}] = 2;
  EXPECT_EQ(tensor[0], 6);
}

TEST(BroadcastTest, TwoTensors) {
  Tensor tensor1 = Tensor::range(1, 4).reshape({1, 3});
  Tensor tensor2 = Tensor::range(1, 4).reshape({3, 1});
  auto [b1, b2] = broadcast(tensor1, tensor2);
  EXPECT_EQ(b1.shape(), (array_t{3, 3}));
  EXPECT_EQ(b2.shape(), (array_t{3, 3}));
}

TEST(TruncateTest, Vector) {
  Tensor tensor = Tensor::range(1, 9).reshape({8});
  Tensor truncated = truncate(tensor, {1}, {3});
  EXPECT_EQ(truncated.shape(), array_t{2});
  EXPECT_EQ(truncated, Tensor::range(2, 4).reshape({2}));
}

TEST(TruncateTest, MatrixRows) {
  Tensor tensor = Tensor::range(1, 9).reshape({4, 2});
  Tensor truncated = truncate(tensor, {1}, {3});
  EXPECT_EQ(truncated.shape(), (array_t{2, 2}));
  EXPECT_EQ(truncated, Tensor::range(3, 7).reshape({2, 2}));
}

TEST(TruncateTest, MatrixElems) {
  Tensor tensor = Tensor::range(1, 17).reshape({4, 4});
  Tensor truncated = truncate(tensor, {1, 1}, {3, 3});
  EXPECT_EQ(truncated.shape(), (array_t{2, 2}));
  Tensor expected(array_t{2, 2});
  slice(expected, {0}) = Tensor::range(6, 8);
  slice(expected, {1}) = Tensor::range(10, 12);
  EXPECT_EQ(truncated, expected);
}
