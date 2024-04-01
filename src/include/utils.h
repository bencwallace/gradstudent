#include <cstring>
#include <fstream>

#include "tensor.h"

namespace gradstudent {

Tensor read_image(const std::string &filename);

void write_image(const std::string &filename, const Tensor &image);

} // namespace gradstudent
