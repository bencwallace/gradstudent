#include "tensor.h"

namespace gradstudent {

void addKernel(Tensor &, const Tensor &, const Tensor &);

void multKernel(Tensor &, const double, const Tensor &);

void multKernel(Tensor &, const Tensor &, const Tensor &);

void negKernel(Tensor &, const Tensor &);

void dotKernel(Tensor &, const Tensor &, const Tensor &);

} // namespace gradstudent
