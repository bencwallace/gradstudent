#include <gtest/gtest.h>

#include "meta.h"

using namespace gradstudent;

TEST(BoolToConstTest, BaseCase) {
  EXPECT_TRUE((std::is_same_v<bool_to_const_t<double>, std::tuple<>>));
}

TEST(BoolToConstTest, Rec1) {
  EXPECT_TRUE(
      (std::is_same_v<bool_to_const_t<double, false>, std::tuple<double>>));
  EXPECT_TRUE(
      (std::is_same_v<bool_to_const_t<int, true>, std::tuple<const int>>));
}

TEST(BoolToConstTest, Rec2) {
  EXPECT_TRUE((std::is_same_v<bool_to_const_t<unsigned int, false, false>,
                              std::tuple<unsigned int, unsigned int>>));
  EXPECT_TRUE((std::is_same_v<bool_to_const_t<bool, true, false>,
                              std::tuple<const bool, bool>>));
  EXPECT_TRUE((std::is_same_v<bool_to_const_t<double, false, true>,
                              std::tuple<double, const double>>));
  EXPECT_TRUE((std::is_same_v<bool_to_const_t<double *, true, true>,
                              std::tuple<double *const, double *const>>));
}

TEST(RefAdderTest, RefTest) {
  using T2 = add_ref_t<std::tuple<double, const double, double>>;
  EXPECT_TRUE(
      (std::is_same_v<T2, std::tuple<double &, const double &, double &>>));
}
