#include <gtest/gtest.h>

#include "multi_index.h"

TEST(ToArrayTest, Dim1) {
  MultiIndex mIdx({5});
  mIdx[0] = 13;
  array_t arr((array_t)mIdx);
  EXPECT_EQ(arr.size(), 1);
  EXPECT_EQ(arr[0], 13);
}
