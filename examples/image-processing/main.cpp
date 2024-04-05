#include <iostream>
#include <memory>

#include "gradstudent.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <image.pgm>" << std::endl;
    return 1;
  }
  std::string filename(argv[1]);
  auto image = gs::read_image(filename);
  gs::Tensor kernel(gs::array_t{3, 3});
  for (auto [idx, x] : gs::ITensorIter(kernel)) {
    if (idx == gs::array_t{1, 1}) {
      x = 8;
    } else {
      x = -1;
    }
  }
  gs::Tensor out(gs::conv(image, kernel));
  for (const auto &[val] : gs::TensorIter(out)) {
    val = std::max(0.0, std::min(255.0, val));
  }
  gs::write_image(filename + ".out.pgm", out);

  return 0;
}
