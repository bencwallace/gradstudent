#include <gtest/gtest.h>

#include "tensor.h"

TEST(CtorsTest, Empty) {
  Tensor t1({3, 4, 5});
  EXPECT_EQ(t1.ndims(), 3);
  EXPECT_EQ(t1.size(), 3 * 4 * 5);
  EXPECT_EQ(t1.shape(), Array({3, 4, 5}));
  EXPECT_EQ(t1.strides(), Array({20, 5, 1}));
}

TEST(CtorsTest, NonEmpty) {
    std::initializer_list<double> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    Tensor t1({3, 4}, {1, 3}, data);
    EXPECT_EQ(t1.ndims(), 2);
    EXPECT_EQ(t1.size(), 12);
    EXPECT_EQ(t1.shape(), Array({3, 4}));
    EXPECT_EQ(t1.strides(), Array({1, 3}));
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(t1[i], i);
    }
}

TEST(CtorsTest, NonEmptyDefaultStrides) {
    std::initializer_list<double> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    Tensor t1({3, 4}, data);
    EXPECT_EQ(t1.ndims(), 2);
    EXPECT_EQ(t1.size(), 12);
    EXPECT_EQ(t1.shape(), Array({3, 4}));
    EXPECT_EQ(t1.strides(), Array({4, 1}));
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(t1[i], i);
    }
}

TEST(CtorsTest, Scalar) {
    Tensor t1(3.14);
    EXPECT_EQ(t1.ndims(), 0);
    EXPECT_EQ(t1.size(), 1);
    EXPECT_EQ(t1.shape(), Array{});
    EXPECT_EQ(t1.strides(), Array{});
    EXPECT_EQ(t1[0], 3.14);
}

TEST(CtorsTest, Copy) {
    // initialize tensor with *non-default* strides
    Tensor t1({2, 2}, {1, 2}, {1, 3, 2, 4});

    // copy should have same shape
    Tensor t2(t1);
    EXPECT_EQ(t2.ndims(), 2);
    EXPECT_EQ(t2.size(), 4);
    EXPECT_EQ(t2.shape(), Array({2, 2}));

    // copy should have *default* strides
    EXPECT_EQ(t2.strides(), Array({2, 1}));
    // copy data should be re-arranged to match new strides
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(t2[i], i + 1);
    }

    // copy should not act as a view
    t2[0] = 0;
    EXPECT_EQ(t1[0], 1);
}

TEST(CtorsTest, View) {
    Tensor t1({2, 2}, {1, 2}, {1, 3, 2, 4});
    Tensor t2({2, 2}, {2, 1}, t1, 0);
    EXPECT_EQ(t2.ndims(), 2);
    EXPECT_EQ(t2.size(), 4);
    EXPECT_EQ(t2.shape(), Array({2, 2}));
    EXPECT_EQ(t2.strides(), Array({2, 1}));

    // modifications to view should change underlying tensor
    t2[0] = 0;
    EXPECT_EQ(t1[0], 0);
}
