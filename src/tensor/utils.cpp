#include "tensor.h"

void Tensor::ensureWritable() {
  // implements copy-on-write
  // should be called prior to any write operation
  if (ro_) {
    double *temp = new double[size_];
    std::memcpy(temp, data_.get(), size_ * sizeof(double));
    data_.reset(temp);
    ro_ = false;
  }
}
