#include <fstream>
#include <sstream>

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
  size_t width = 0;
  size_t height = 0;
  size_t depth = 0;
  ss << file.rdbuf();
  ss >> width >> height;
  ss >> depth;

  Tensor result(array_t{height, width});
  unsigned char x = 0;
  for (const auto &[val] : TensorIter(result)) {
    // TODO: cast directly
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
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

  file << "P5\n";
  file << width << " " << height << '\n';
  file << "255\n";
  for (const auto &[val] : TensorIter(image)) {
    // TODO: cast directly
    auto temp = static_cast<unsigned char>(val);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    file.write(reinterpret_cast<const char *>(&temp), sizeof(char));
  }
}

} // namespace gs
