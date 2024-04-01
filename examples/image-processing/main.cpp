#include <iostream>
#include <memory>

#include "gradstudent.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <image.pgm>" << std::endl;
    return 1;
  }
  std::string filename(argv[1]);
  auto image = gradstudent::read_image(filename);
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
  gradstudent::write_image(filename + ".out.pgm", out);

  return 0;
}
