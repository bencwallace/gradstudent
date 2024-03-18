#include <gtest/gtest.h>

#include "utils.h"

using namespace gradstudent;

TEST(BroadcastShapesTest, Fail_0) {
  array_t out_shape;

  const array_t left_shape = {2};
  const array_t right_shape = {3};

  EXPECT_THROW(broadcastShapes(out_shape, left_shape, right_shape),
               std::invalid_argument);
}

TEST(BroadcastShapesTest, Fail_1) {
  array_t out_shape;

  const array_t left_shape = {1, 2};
  const array_t right_shapes[] = {{3}, {1, 3}};
  for (const auto &right_shape : right_shapes) {
    EXPECT_THROW(broadcastShapes(out_shape, left_shape, right_shape),
                 std::invalid_argument);
  }
}

TEST(BroadcastShapesTest, Fail_2) {
  array_t out_shape;

  const array_t left_shape = {4, 2};
  const array_t right_shapes[] = {{3}, {3, 2}, {2, 2}, {2, 1}};
  for (const auto &right_shape : right_shapes) {
    EXPECT_THROW(broadcastShapes(out_shape, left_shape, right_shape),
                 std::invalid_argument);
  }
}

TEST(BroadcastShapesTest, Fail_3) {
  array_t out_shape;

  const array_t left_shape = {5, 4, 2};
  const array_t right_shapes[] = {{5, 3, 2}, {1, 4, 3}, {5, 2, 1}};
  for (const auto &right_shape : right_shapes) {
    EXPECT_THROW(broadcastShapes(out_shape, left_shape, right_shape),
                 std::invalid_argument);
  }
}

TEST(BroadcastShapesTest, Succeed_0) {
  array_t out_shape;

  const array_t left_shape = {3};
  const array_t right_shapes[] = {{1}, {3}, {1, 1}, {1, 3}, {3, 1}, {4, 3}};
  for (const auto &right_shape : right_shapes) {
    EXPECT_NO_THROW(broadcastShapes(out_shape, left_shape, right_shape));
  }
}

TEST(BroadcastShapesTest, Succeed_1) {
  array_t out_shape;

  const array_t left_shape = {3, 1};
  const array_t right_shapes[] = {{3}, {1, 3}, {3, 1}, {2, 3, 2}};
  for (const auto &right_shape : right_shapes) {
    EXPECT_NO_THROW(broadcastShapes(out_shape, left_shape, right_shape));
  }
}

TEST(BroadcastShapesTest, Succeed_2) {
  array_t out_shape;

  const array_t left_shape = {3, 2};
  const array_t right_shapes[] = {{2}, {1, 2}, {3, 2}, {4, 1, 2}};
  for (const auto &right_shape : right_shapes) {
    EXPECT_NO_THROW(broadcastShapes(out_shape, left_shape, right_shape));
  }
}

TEST(BroadcastShapesTest, MaskRight) {
  array_t out_shape;

  const array_t left_shape = {3, 2};
  const array_t right_shapes[] = {{1}, {1, 1}};
  for (const auto &right_shape : right_shapes) {
    EXPECT_EQ(broadcastShapes(out_shape, left_shape, right_shape),
              std::vector<int>({BCAST_RIGHT, BCAST_RIGHT}));
  }
}

TEST(BroadcastShapesTest, MaskLeft) {
  array_t out_shape;

  const array_t left_shapes[] = {{1}, {1, 1}};
  const array_t right_shape = {4, 3, 2};
  for (const auto &left_shape : left_shapes) {
    EXPECT_EQ(broadcastShapes(out_shape, left_shape, right_shape),
              std::vector<int>({BCAST_LEFT, BCAST_LEFT, BCAST_LEFT}));
  }
}

TEST(BroadcastShapesTest, MaskMixed) {
  array_t out_shape;

  const array_t left_shape = {6, 1, 4, 3, 1};
  const array_t right_shape = {1, 1, 3, 3};
  EXPECT_EQ(broadcastShapes(out_shape, left_shape, right_shape),
            std::vector<int>({BCAST_RIGHT, BCAST_NONE, BCAST_RIGHT, BCAST_NONE,
                              BCAST_LEFT}));
}
