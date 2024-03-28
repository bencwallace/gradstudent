#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

#include "gradstudent.h"

gradstudent::Tensor read_image(const std::string &filename) {
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

  gradstudent::Tensor result(gradstudent::array_t{height, width});
  unsigned char x;
  for (const auto &[val] : gradstudent::TensorIter(result)) {
    // TODO: cast directly
    ss.read(reinterpret_cast<char *>(&x), sizeof(x));
    val = static_cast<double>(x);
  }

  return result;
}

void write_image(const std::string &filename,
                 const gradstudent::Tensor &image) {
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
  for (const auto &[val] : gradstudent::TensorIter(image)) {
    // TODO: cast directly
    auto temp = static_cast<unsigned char>(val);
    file.write(reinterpret_cast<const char *>(&temp), sizeof(char));
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <image.pgm>" << std::endl;
    return 1;
  }
  std::string filename(argv[1]);
  auto image = read_image(filename);
  gradstudent::Tensor kernel(gradstudent::array_t{3, 3});
  for (auto [idx, x] : gradstudent::ITensorIter(kernel)) {
    if (idx == gradstudent::array_t{1, 1}) {
      x = 8;
    } else {
      x = -1;
    }
  }
  gradstudent::Tensor out(gradstudent::conv(image, kernel));
  for (const auto &[val] : gradstudent::TensorIter(out)) {
    val = std::max(0.0, std::min(255.0, val));
  }
  write_image(filename + ".out.pgm", out);

  return 0;
}
