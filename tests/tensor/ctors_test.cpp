#include <gtest/gtest.h>

#include "gradstudent/tensor.h"

using namespace gs;

TEST(CtorsTest, Empty) {
  Tensor t1({3, 4, 5});
  EXPECT_EQ(t1.ndims(), 3);
  EXPECT_EQ(t1.size(), 3 * 4 * 5);
  EXPECT_EQ(t1.shape(), array_t({3, 4, 5}));
  EXPECT_EQ(t1.strides(), array_t({20, 5, 1}));
}

TEST(CtorsTest, Scalar) {
  Tensor t1(3.14);
  EXPECT_EQ(t1.ndims(), 0);
  EXPECT_EQ(t1.size(), 1);
  EXPECT_EQ(t1.shape(), array_t{});
  EXPECT_EQ(t1.strides(), array_t{});
  EXPECT_EQ(t1[0], 3.14);
}

TEST(CtorsTest, Copy) {
  // initialize tensor with *non-default* strides
  Tensor t1 = Tensor::range(1, 5).reshape({2, 2}, {1, 2});

  // copy should have same shape
  Tensor t2(t1);
  EXPECT_EQ(t2.ndims(), 2);
  EXPECT_EQ(t2.size(), 4);
  EXPECT_EQ(t2.shape(), array_t({2, 2}));

  // copy should have *default* strides
  EXPECT_EQ(t2.strides(), array_t({2, 1}));
  // copy data should be re-arranged to match new strides
  EXPECT_EQ((t2[{0, 0}]), 1);
  EXPECT_EQ((t2[{0, 1}]), 3);
  EXPECT_EQ((t2[{1, 0}]), 2);
  EXPECT_EQ((t2[{1, 1}]), 4);

  // copy should not act as a view
  t2[0] = 0;
  EXPECT_EQ(t1[0], 1);
}

TEST(CtorsTest, View) {
  Tensor t1 = Tensor::range(1, 5).reshape({2, 2}, {1, 2});
  Tensor t2({2, 2}, {2, 1}, t1, 0);
  EXPECT_EQ(t2.ndims(), 2);
  EXPECT_EQ(t2.size(), 4);
  EXPECT_EQ(t2.shape(), array_t({2, 2}));
  EXPECT_EQ(t2.strides(), array_t({2, 1}));

  // modifications to view should change underlying tensor
  t2[0] = 0;
  EXPECT_EQ(t1[0], 0);
}
