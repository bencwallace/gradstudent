#include "tensor.h"
#include "tensor_iter.h"

namespace gradstudent {

void Tensor::ensureWritable() {
  // implements copy-on-write
  // should be called prior to any write operation
  if (ro_) {
    double *temp = new double[size_];
    size_t i = 0;
    for (const auto &[val] : TensorIter(*this)) {
      temp[i++] = val;
    }
    data_.reset(temp);
    ro_ = false;
    offset_ = 0;
    strides_ = defaultStrides(shape_);
  }
}

} // namespace gradstudent
