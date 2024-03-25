#include <gtest/gtest.h>

#include "meta.h"

using namespace gradstudent;

TEST(ConstConvertTest, BaseCase) {
  EXPECT_TRUE((std::is_same_v<const_convert_t<double>, std::tuple<>>));
}

TEST(ConstConvertTest, Rec1) {
  EXPECT_TRUE(
      (std::is_same_v<const_convert_t<double, int>, std::tuple<double>>));
  EXPECT_TRUE((
      std::is_same_v<const_convert_t<int, const char>, std::tuple<const int>>));
}

TEST(ConstConvertTest, Rec2) {
  EXPECT_TRUE((std::is_same_v<const_convert_t<unsigned int, int, char>,
                              std::tuple<unsigned int, unsigned int>>));
  EXPECT_TRUE((std::is_same_v<const_convert_t<bool, const int, const char *>,
                              std::tuple<const bool, bool>>));
  EXPECT_TRUE(
      (std::is_same_v<const_convert_t<double, const bool &, double *const>,
                      std::tuple<double, const double>>));
  EXPECT_TRUE(
      (std::is_same_v<const_convert_t<double *, const char *const, const int>,
                      std::tuple<double *const, double *const>>));
}

TEST(ConstConvertTest, RefTest) {
  using T2 = add_ref_t<std::tuple<double, const double, double>>;
  EXPECT_TRUE(
      (std::is_same_v<T2, std::tuple<double &, const double &, double &>>));
}
