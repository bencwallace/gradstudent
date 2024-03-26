#include <gtest/gtest.h>

#include "tensor.h"
#include "tensor_iter.h"

using namespace gradstudent;

TEST(TensorIterTest, DefaultStrides) {
  Tensor t1({2, 2}, {1, 2, 3, 4});
  const Tensor t2({2, 2}, {4, 3, 2, 1});
  TensorIter mit(t1, t2);

  for (auto vals : TensorIter(t1, t2)) {
    static_assert(std::tuple_size_v<decltype(vals)> == 2);
    auto [x, y] = vals;
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
  Tensor t1({2, 2}, {2, 1}, {1, 2, 3, 4});
  const Tensor t2({2, 2}, {1, 2}, {4, 2, 3, 1});

  for (auto vals : TensorIter(t1, t2)) {
    static_assert(std::tuple_size_v<decltype(vals)> == 2);
    auto [x, y] = vals;
    EXPECT_EQ(x + y, 5);
  }

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