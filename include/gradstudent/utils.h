/**
 * @file utils.h
 * @author Ben Wallace (me@bcwallace.com)
 * @brief Utility functions
 * @version 0.1
 * @date 2024-04-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include <array>
#include <cstring>
#include <istream>

#include "gradstudent/tensor.h"
#include "gradstudent/internal/utils.h"

namespace gs {

// @cond
template <typename T>
Tensor parse_numpy_data(const array_t &shape, std::istream &file) {
  Tensor result(shape);
  std::array<unsigned char, sizeof(T)> data;
  T val;
  for (size_t i = 0; i < prod(shape); ++i) {
    file.read(reinterpret_cast<char *>(data.data()), sizeof(T)); // NOLINT
    std::memcpy(&val, data.data(), sizeof(T));
    result[i] = static_cast<double>(val);
  }
  return result;
}
// @endcond

/**
 * @brief Reads a file in the NPY format
 *
 * Currently, the following restrictions apply:
 *
 *   * Format version must be 1.0.
 *   * `fortran_order` must be false (only C order supported).
 *   * Array data may not contain Python objects.
 *   * Array dtype must be `<f8` (little-endian 64-bit floating point).
 *
 * For more information: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
 *
 * @param filename File to parse
 * @return Tensor The tensor parsed out of the file
 */
Tensor read_numpy(const std::string &filename);

/**
 * @brief Reads a PGM image into a tensor
 *
 * For more information: https://netpbm.sourceforge.net/doc/pgm.html
 *
 * @param filename The image to read
 * @return Tensor The corresponding tensor
 */
Tensor read_image(const std::string &filename);

/**
 * @brief Writes a tensor to an image in PGM format
 *
 * For more information: https://netpbm.sourceforge.net/doc/pgm.html
 *
 * @param filename The output path
 * @param image The tensor to write
 */
void write_image(const std::string &filename, const Tensor &image);

} // namespace gs
