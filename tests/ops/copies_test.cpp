#include <gtest/gtest.h>

#include "ops.h"
#include "tensor.h"

using namespace gradstudent;

TEST(FlattenTest, Flatten) {
  const Tensor matrix({2, 2}, {1, 2, 3, 4});
  Tensor flat = flatten(matrix);
  ASSERT_EQ(flat.shape(), array_t{4});
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(flat[i], i + 1);
  }
}
