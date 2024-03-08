#include "tensor.h"

void addKernel(Tensor &, const Tensor &, const Tensor &);

void multKernel(Tensor &, const double, const Tensor &);

void multKernel(Tensor &, const Tensor &, const Tensor &);

void negKernel(Tensor &, const Tensor &);

void dotKernel(Tensor &, const Tensor &, const Tensor &);

double norm2(const Tensor &);
