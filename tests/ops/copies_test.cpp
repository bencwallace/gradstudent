#include <gtest/gtest.h>

#include "ops.h"
#include "tensor.h"

using namespace gs;

TEST(FlattenTest, Flatten) {
  const Tensor matrix = Tensor::range(1, 5).reshape({2, 2});
  Tensor flat = flatten(matrix);
  ASSERT_EQ(flat.shape(), array_t{4});
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(flat[i], i + 1);
  }
}
