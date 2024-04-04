#include <array>
#include <cstring>
#include <fstream>

#include "tensor.h"

namespace gradstudent {

template <typename T>
Tensor parse_numpy_data(const array_t &shape, std::istream &file) {
  Tensor result(shape);
  std::array<unsigned char, sizeof(T)> data;
  T val;
  for (size_t i = 0; i < prod(shape); ++i) {
    file.read(reinterpret_cast<char *>(data.data()),
              sizeof(T)); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    std::memcpy(&val, data.data(), sizeof(T));
    result[i] = static_cast<double>(val);
  }
  return result;
}

Tensor read_numpy(const std::string &filename);

Tensor read_image(const std::string &filename);

void write_image(const std::string &filename, const Tensor &image);

} // namespace gradstudent
