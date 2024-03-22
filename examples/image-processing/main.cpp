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
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ss.read(reinterpret_cast<char *>(&x), sizeof(x));
      result[{i, j}] = static_cast<double>(x);
    }
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
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      auto temp = static_cast<unsigned char>(image[{i, j}]);
      file.write(reinterpret_cast<const char *>(&temp), sizeof(char));
    }
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
  for (size_t i = 0; i < 9; ++i) {
    if (i == 4) {
      kernel[i] = 8;
    } else {
      kernel[i] = -1;
    }
  }
  gradstudent::Tensor out(gradstudent::conv(image, kernel));
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = std::max(0.0, std::min(255.0, out[i]));
  }
  write_image(filename + ".out.pgm", out);

  return 0;
}
