#include "tensor.h"

namespace gradstudent {

void Tensor::ensureWritable() {
  // implements copy-on-write
  // should be called prior to any write operation
  if (ro_) {
    double *temp = new double[size_];
    size_t i = 0;
    for (auto mIdx : multiIndexRange()) {
      temp[i++] = data_[toIndex(mIdx)];
    }
    data_.reset(temp);
    ro_ = false;
  }
}

} // namespace gradstudent
