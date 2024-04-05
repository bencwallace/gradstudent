#include <iostream>

#include "gradstudent.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::cerr << "Usage: " << argv[0] << " <image.pgm>\n";
    return 1;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::string filename(argv[1]);
  auto image = gs::read_image(filename);
  size_t kernel_size = 3;
  gs::Tensor kernel(gs::array_t{kernel_size, kernel_size});
  for (auto [idx, x] : gs::ITensorIter(kernel)) {
    if (idx == gs::array_t{1, 1}) {
      x = kernel_size * kernel_size - 1;
    } else {
      x = -1;
    }
  }
  gs::Tensor out(gs::conv(image, kernel));
  for (const auto &[val] : gs::TensorIter(out)) {
    val = std::max(0.0, std::min(255.0, val)); // NOLINT
  }
  gs::write_image(filename + ".out.pgm", out);

  return 0;
}
