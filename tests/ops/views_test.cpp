#include <gtest/gtest.h>

#include "ops.h"
#include "tensor.h"

using namespace gradstudent;

TEST(SliceTest, GetSlice) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor sliced = slice(matrix1, array_t{0});
  EXPECT_EQ(sliced.shape(), array_t({2}));
  EXPECT_EQ(sliced, Tensor({2}, {1, 2}));
  sliced[0] = 0;
  EXPECT_EQ(matrix1[0], 0);
  EXPECT_EQ(matrix1[1], 2);
  EXPECT_EQ(matrix1[2], 3);
  EXPECT_EQ(matrix1[3], 4);
}

TEST(SliceTest, SetSlice) {
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor vector1({2}, {1}, {5, 6});
  slice(matrix1, array_t{0}) = vector1;
  EXPECT_EQ(matrix1[0], 5);
  EXPECT_EQ(matrix1[1], 6);
  EXPECT_EQ(matrix1[2], 3);
  EXPECT_EQ(matrix1[3], 4);
}

TEST(SliceTest, GetSliceConst) {
  const Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor sliced = slice(matrix1, array_t{0});
  sliced[0] = 0;
  EXPECT_EQ(matrix1[0], 1);
}

TEST(SliceTest, SetSliceConst) {
  const Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor vector1({2}, {1}, {5, 6});
  Tensor sliced = slice(matrix1, array_t{0});
  sliced = vector1;
  EXPECT_EQ(matrix1[0], 1);
  EXPECT_EQ(matrix1[1], 2);
}

TEST(FlattenTest, Subscript) {
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  Tensor flat = flatten(matrix);

  EXPECT_EQ(flat.ndims(), 1);
  EXPECT_EQ(flat.shape(), array_t{4});

  flat[0] = 4;
  flat[1] = 3;
  flat[2] = 2;
  flat[3] = 1;
  EXPECT_EQ(matrix[0], 4);
  EXPECT_EQ(matrix[1], 3);
  EXPECT_EQ(matrix[2], 2);
  EXPECT_EQ(matrix[3], 1);
}

TEST(FlattenTest, Assign) {
  Tensor matrix({2, 2}, {1, 2, 3, 4});
  Tensor vector({4}, {4, 3, 2, 1});
  flatten(matrix) = vector;
  EXPECT_EQ(matrix[0], 4);
  EXPECT_EQ(matrix[1], 3);
  EXPECT_EQ(matrix[2], 2);
  EXPECT_EQ(matrix[3], 1);
}

TEST(FlattenTest, SubscriptConst) {
  const Tensor matrix({2, 2}, {1, 2, 3, 4});

  // const tensor returned by flatten can be assigned to non-const tensor...
  Tensor flat = flatten(matrix);
  // ...but on wrote this non-const tensor detaches itself (internally: makes a
  // copy)
  flat[0] = 0;
  // original tensor is unchanged
  EXPECT_EQ(matrix[0], 1);
}

TEST(FlattenTest, AssignConst) {
  const Tensor matrix({2, 2}, {1, 2, 3, 4});
  Tensor vector({4}, {4, 3, 2, 1});
  Tensor flat = flatten(matrix);
  flat = vector;
  EXPECT_EQ(matrix[0], 1);
  EXPECT_EQ(matrix[1], 2);
  EXPECT_EQ(matrix[2], 3);
  EXPECT_EQ(matrix[3], 4);
}

TEST(PermuteTest, Subscript) {
  Tensor matrix({2, 3}, {1, 2, 3, 4, 5, 6});
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
  Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {5, 6, 7, 8});
  permute(matrix1, {1, 0}) = matrix2;
  EXPECT_EQ((matrix1[{0, 0}]), 5);
  EXPECT_EQ((matrix1[{0, 1}]), 7);
  EXPECT_EQ((matrix1[{1, 0}]), 6);
  EXPECT_EQ((matrix1[{1, 1}]), 8);
}

TEST(PermuteTest, SubscriptConst) {
  const Tensor matrix({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor perm = permute(matrix, {1, 0});
  perm[0] = 0;
  EXPECT_EQ(matrix[0], 1);
}

TEST(PermuteTest, AssignConst) {
  const Tensor matrix1({2, 2}, {1, 2, 3, 4});
  Tensor matrix2({2, 2}, {5, 6, 7, 8});
  Tensor perm = permute(matrix1, {1, 0});
  perm = matrix2;
  EXPECT_EQ((matrix1[{0, 0}]), 1);
  EXPECT_EQ((matrix1[{0, 1}]), 2);
  EXPECT_EQ((matrix1[{1, 0}]), 3);
  EXPECT_EQ((matrix1[{1, 1}]), 4);
}
