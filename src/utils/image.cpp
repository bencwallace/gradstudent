#include <cstring>
#include <fstream>

#include "internal/utils.h"
#include "tensor.h"
#include "tensor_iter.h"

namespace gs {

Tensor read_image(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::in);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  std::string line;

  std::getline(file, line);
  if (line != "P5") {
    throw std::runtime_error("Unsupported image format. Header: " + line);
  }

  std::stringstream ss;
  size_t width, height, depth;
  ss << file.rdbuf();
  ss >> width >> height;
  ss >> depth;

  Tensor result(array_t{height, width});
  unsigned char x;
  for (const auto &[val] : TensorIter(result)) {
    // TODO: cast directly
    ss.read(reinterpret_cast<char *>(&x), sizeof(x));
    val = static_cast<double>(x);
  }

  return result;
}

void write_image(const std::string &filename, const Tensor &image) {
  std::ofstream file(filename,
                     std::ios::binary | std::ios::out | std::ios::trunc);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  size_t height = image.shape()[0];
  size_t width = image.shape()[1];

  file << "P5" << std::endl;
  file << width << " " << height << std::endl;
  file << "255" << std::endl;
  for (const auto &[val] : TensorIter(image)) {
    // TODO: cast directly
    auto temp = static_cast<unsigned char>(val);
    file.write(reinterpret_cast<const char *>(&temp), sizeof(char));
  }
}

} // namespace gs
