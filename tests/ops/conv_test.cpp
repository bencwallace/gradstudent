#include "gradstudent/iter.h"
#include "gradstudent/ops.h"
#include "gradstudent/tensor.h"

#include <gtest/gtest.h>

using namespace gs;

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

TEST(ConvTest, 1DOnesFilters) {
  for (size_t input_size = 2; input_size < 8; ++input_size) {
    Tensor input = Tensor::fill(array_t{input_size}, 1);
    for (size_t numFilters = 1; numFilters < 4; ++numFilters) {
      for (size_t kernel_size = 1; kernel_size < input_size; ++kernel_size) {
        Tensor kernel = Tensor::fill(array_t{numFilters, kernel_size}, 1);
        Tensor output = conv(input, kernel);
        ASSERT_EQ(output.shape(),
                  (array_t{numFilters, input_size - kernel_size + 1}))
            << "input_size: " << input_size << ", kernel_size: " << kernel_size
            << ", numFilters: " << numFilters;
        for (size_t i = 0; i < output.size(); ++i) {
          EXPECT_EQ(output[i], kernel_size)
              << "input_size: " << input_size
              << ", kernel_size: " << kernel_size << ", i: " << i
              << ", numFilters: " << numFilters;
        }
      }
    }
  }
}

TEST(ConvTest, 2DOnesSquareFilters) {
  for (size_t input_size = 2; input_size < 8; ++input_size) {
    Tensor input = Tensor::fill(array_t{input_size, input_size}, 1);
    for (size_t numFilters = 1; numFilters < 4; ++numFilters) {
      for (size_t kernel_size = 1; kernel_size < input_size; ++kernel_size) {
        Tensor kernel =
            Tensor::fill(array_t{numFilters, kernel_size, kernel_size}, 1);
        Tensor output = conv(input, kernel);
        ASSERT_EQ(output.shape(),
                  (array_t{numFilters, input_size - kernel_size + 1,
                           input_size - kernel_size + 1}))
            << "input_size: " << input_size << ", kernel_size: " << kernel_size;
        for (size_t i = 0; i < output.size(); ++i) {
          EXPECT_EQ(output[i], kernel_size * kernel_size)
              << "input_size: " << input_size
              << ", kernel_size: " << kernel_size << ", i: " << i;
        }
      }
    }
  }
}

TEST(Conv2dTest, 2DOnesSquare) {
  for (size_t depth = 1; depth < 4; ++depth) {
    for (size_t input_size = 2; input_size < 6; ++input_size) {
      for (size_t kernel_size = 1; kernel_size < input_size; ++kernel_size) {
        Tensor input = Tensor::fill(array_t{input_size, input_size, depth}, 1);
        Tensor kernel =
            Tensor::fill(array_t{kernel_size, kernel_size, depth}, 1);
        Tensor output = conv(input, kernel, 2);
        ASSERT_EQ(output.shape(), (array_t{input_size - kernel_size + 1,
                                           input_size - kernel_size + 1}));
        for (size_t i = 0; i < output.size(); ++i) {
          EXPECT_EQ(output[i], kernel_size * kernel_size * depth);
        }
      }
    }
  }
}

TEST(Conv2dTest, 2DOnesSquareFilters) {
  for (size_t depth = 1; depth < 4; ++depth) {
    for (size_t input_size = 2; input_size < 6; ++input_size) {
      for (size_t num_filters = 1; num_filters < 4; ++num_filters) {
        for (size_t kernel_size = 1; kernel_size < input_size; ++kernel_size) {
          Tensor input =
              Tensor::fill(array_t{input_size, input_size, depth}, 1);
          Tensor kernel = Tensor::fill(
              array_t{num_filters, kernel_size, kernel_size, depth}, 1);
          Tensor output = conv(input, kernel, 2);
          ASSERT_EQ(output.shape(),
                    (array_t{num_filters, input_size - kernel_size + 1,
                             input_size - kernel_size + 1}));
          for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_EQ(output[i], kernel_size * kernel_size * depth);
          }
        }
      }
    }
  }
}

TEST(MaxPoolTest, 1DRange) {
  size_t input_size = 10;
  Tensor input = Tensor::range(input_size);

  auto result = maxPool(input, array_t{2});
  ASSERT_EQ(result.shape(), array_t{input_size / 2});
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(result[i], 2 * i + 1) << "i = " << i;
  }
}

TEST(MaxPoolTest, 2DRange) {
  size_t input_size = 10;
  Tensor input =
      Tensor::range(input_size * input_size).reshape({input_size, input_size});

  auto result = maxPool(input, array_t{2, 5});
  ASSERT_EQ(result.shape(), (array_t{5, 2}));
  for (auto [idx, val] : ITensorIter(result)) {
    EXPECT_EQ(val, 14 + 20 * idx[0] + 5 * idx[1])
        << "i  = " << idx[0] << ", j = " << idx[1];
  }
}

TEST(MaxPoolTest, 2DRangeChannels) {
  size_t input_size = 10;
  size_t num_channels = 4;
  Tensor input(array_t{num_channels, input_size, input_size});
  for (size_t i = 0; i < num_channels; ++i) {
    slice(input, {i}) = Tensor::range(input_size * input_size)
                            .reshape({input_size, input_size});
  }

  auto result = maxPool(input, array_t{2, 5});
  ASSERT_EQ(result.shape(), (array_t{num_channels, 5, 2}));
  for (size_t i = 0; i < num_channels; ++i) {
    const auto &s = slice(result, {i});
    for (auto [idx, val] : ITensorIter(s)) {
      EXPECT_EQ(val, 14 + 20 * idx[0] + 5 * idx[1])
          << "i  = " << idx[0] << ", j = " << idx[1];
    }
  }
}
