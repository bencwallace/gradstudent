#include "ops.h"
#include "tensor.h"

#include <gtest/gtest.h>

using namespace gradstudent;

TEST(ConvTest, 1DOnes) {
  for (size_t input_size = 2; input_size < 8; ++input_size) {
    Tensor input = Tensor::fill(array_t{input_size}, 1);
    for (size_t kernel_size = 1; kernel_size < input_size; ++kernel_size) {
      Tensor kernel = Tensor::fill(array_t{kernel_size}, 1);
      Tensor output = conv(input, kernel);
      ASSERT_EQ(output.shape(), array_t{input_size - kernel_size + 1})
          << "input_size: " << input_size << ", kernel_size: " << kernel_size;
      for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], kernel_size)
            << "input_size: " << input_size << ", kernel_size: " << kernel_size
            << ", i: " << i;
      }
    }
  }
}

TEST(ConvTest, 2DOnesSquare) {
  for (size_t input_size = 2; input_size < 8; ++input_size) {
    Tensor input = Tensor::fill(array_t{input_size, input_size}, 1);
    for (size_t kernel_size = 1; kernel_size < input_size; ++kernel_size) {
      Tensor kernel = Tensor::fill(array_t{kernel_size, kernel_size}, 1);
      Tensor output = conv(input, kernel);
      ASSERT_EQ(output.shape(), (array_t{input_size - kernel_size + 1,
                                         input_size - kernel_size + 1}))
          << "input_size: " << input_size << ", kernel_size: " << kernel_size;
      for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], kernel_size * kernel_size)
            << "input_size: " << input_size << ", kernel_size: " << kernel_size
            << ", i: " << i;
      }
    }
  }
}
