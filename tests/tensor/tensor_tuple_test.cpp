#include <gtest/gtest.h>

#include "tensor.h"
#include "tensor_iter.h"

using namespace gradstudent;

TEST(TensorTupleTest, TensorTupleTest) {
  const Tensor tensor1({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor tensor2({2, 3}, {6, 5, 4, 3, 2, 1});
  TensorTuple tensor_tuple(tensor1, tensor2);

  for (auto vals : tensor_tuple) {
    auto [val1, val2] = vals;
    EXPECT_EQ(val1 + val2, 7);
    val2 = 2;
  }
  for (auto mIdx : MultiIndexIter(tensor2.shape())) {
    EXPECT_EQ(tensor2[mIdx], 2);
  }
}
