#include "tensor.h"

void addOp(Tensor &, const Tensor &, const Tensor &);

void multOp(Tensor &, const double, const Tensor &);

void multOp(Tensor &, const Tensor &, const Tensor &);

void negOp(Tensor &, const Tensor &);

void dotOp(Tensor &, const Tensor &, const Tensor &);

double norm2(const Tensor &);
