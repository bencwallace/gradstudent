#include <gtest/gtest.h>

#include "array.h"

TEST(ArrayTest, PrintScalarTest) {
  Array array1({10});
  std::stringstream ss;
  ss << array1;
  EXPECT_EQ(ss.str(), "(10,)");
}

TEST(ArrayTest, PrintTest) {
  Array array2({1, 2, 3});
  std::stringstream ss;
  ss << array2;
  EXPECT_EQ(ss.str(), "(1, 2, 3)");
}