#include <gtest/gtest.h>

#include "internal/meta.h"

using namespace gs;

TEST(TupleCatTest, Empty) {
  static_assert(std::is_same_v<tuple_cat_t<std::tuple<>>, std::tuple<>>);
}

TEST(TupleCatTest, EmptyCatEmpty) {
  static_assert(
      std::is_same_v<tuple_cat_t<std::tuple<>, std::tuple<>>, std::tuple<>>);
}

TEST(TupleCatTest, EmptyCatNonempty) {
  static_assert(std::is_same_v<tuple_cat_t<std::tuple<>, std::tuple<int>>,
                               std::tuple<int>>);
}

TEST(TupleCatTest, NonemptyCatNonempty) {
  static_assert(std::is_same_v<
                tuple_cat_t<std::tuple<int, double>, std::tuple<char, char>>,
                std::tuple<int, double, char, char>>);
}

TEST(TupleTest, BaseCase) {
  static_assert(std::is_same_v<ntuple_t<0, double>, std::tuple<>>);
}

TEST(TupleTest, RecCase) {
  static_assert(std::is_same_v<ntuple_t<4, double>,
                               std::tuple<double, double, double, double>>);
  static_assert(
      std::is_same_v<ntuple_t<4, double &>,
                     std::tuple<double &, double &, double &, double &>>);
}

TEST(BoolToConstTest, BaseCase) {
  static_assert(std::is_same_v<bool_to_const_t<double>, std::tuple<>>);
}

TEST(BoolToConstTest, Rec1) {
  static_assert(
      std::is_same_v<bool_to_const_t<double, false>, std::tuple<double>>);
  static_assert(
      std::is_same_v<bool_to_const_t<int, true>, std::tuple<const int>>);
}

TEST(BoolToConstTest, Rec2) {
  static_assert(std::is_same_v<bool_to_const_t<unsigned int, false, false>,
                               std::tuple<unsigned int, unsigned int>>);
  static_assert(std::is_same_v<bool_to_const_t<bool, true, false>,
                               std::tuple<const bool, bool>>);
  static_assert(std::is_same_v<bool_to_const_t<double, false, true>,
                               std::tuple<double, const double>>);
  static_assert(std::is_same_v<bool_to_const_t<double *, true, true>,
                               std::tuple<double *const, double *const>>);
}

TEST(RefAdderTest, RefTest) {
  using T2 = add_ref_t<std::tuple<double, const double, double>>;
  static_assert(
      std::is_same_v<T2, std::tuple<double &, const double, double &>>);
}
