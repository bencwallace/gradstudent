#include <gtest/gtest.h>

#include "gradstudent/iter.h"
#include "gradstudent/tensor.h"

using namespace gs;

TEST(TensorIterTest, DefaultStrides) {
  Tensor t1 = Tensor::range(1, 5).reshape({2, 2});
  const Tensor t2 = Tensor::range(4, 0, -1).reshape({2, 2});
  TensorIter mit(t1, t2);

  for (auto vals : TensorIter(t1, t2)) {
    static_assert(std::tuple_size_v<decltype(vals)> == 2);
    auto [x, y] = vals;
    static_assert(std::is_const_v<std::remove_reference_t<decltype(y)>>);
    EXPECT_EQ(x + y, 5);
  }

  for (auto vals : TensorIter(t1, t2)) {
    auto [x, y] = vals;
    x = -y;
  }
  for (size_t i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(t1[i], -t2[i]);
  }
}

TEST(TensorIterTest, Strided) {
  Tensor t1 = Tensor::range(1, 5).reshape({2, 2}, {2, 1});           //  1 2 3 4
  const Tensor t2 = Tensor::range(4, 0, -1).reshape({2, 2}, {1, 2}); // 4 2 3 1

  for (auto vals : TensorIter(t1, t2)) {
    auto [x, y] = vals;
    x = -y;
  }

  TensorIter mit(t1);
  for (auto it = mit.begin(); it != mit.end(); ++it) {
    auto [x] = *it;
    EXPECT_EQ(x, -t2[it.index()]);
  }
}

TEST(ITensorIterTest, DefaultStrides) {
  Tensor t1 = Tensor::range(1, 5).reshape({2, 2});
  const Tensor t2 = Tensor::range(4, 0, -1).reshape({2, 2});
  TensorIter mit(t1, t2);

  for (auto vals : ITensorIter(t1, t2)) {
    static_assert(std::tuple_size_v<decltype(vals)> == 3);
    auto [i, x, y] = vals;
    EXPECT_EQ(t1[i], x);
    EXPECT_EQ(t2[i], y);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(y)>>);
    EXPECT_EQ(x + y, 5);
  }

  for (auto vals : ITensorIter(t1, t2)) {
    auto [i, x, y] = vals;
    t1[i] = -t2[i];
  }
  for (size_t i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(t1[i], -t2[i]);
  }
}

TEST(ITensorIterTest, Strided) {
  Tensor t1 = Tensor::range(1, 5).reshape({2, 2}, {2, 1});
  const Tensor t2 = Tensor::range(4, 0, -1).reshape({2, 2}, {1, 2});

  for (auto vals : ITensorIter(t1, t2)) {
    static_assert(std::tuple_size_v<decltype(vals)> == 3);
    auto [i, x, y] = vals;
    EXPECT_EQ(t1[i], x);
    EXPECT_EQ(t2[i], y);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(y)>>);
  }

  for (auto vals : ITensorIter(t1, t2)) {
    auto [i, x, y] = vals;
    t1[i] = -t2[i];
  }

  TensorIter mit(t1);
  for (auto it = mit.begin(); it != mit.end(); ++it) {
    auto [x] = *it;
    EXPECT_EQ(x, -t2[it.index()]);
  }
}
