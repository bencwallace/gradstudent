#include "gradstudent/internal/utils.h"
#include "gradstudent/iter.h"
#include "gradstudent/tensor.h"

namespace gs {

void Tensor::ensureWritable() {
  // implements copy-on-write
  // should be called prior to any write operation
  if (ro_) {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    double *temp = new double[size_];
    size_t i = 0;
    for (const auto &[val] : TensorIter(*this)) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      temp[i++] = val;
    }
    data_.reset(temp);
    ro_ = false;
    offset_ = 0;
    strides_ = defaultStrides(shape_);
  }
}

} // namespace gs
