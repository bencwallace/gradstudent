#include "tensor.h"

Tensor dot(const Tensor &, const Tensor &);

Tensor flatten(const Tensor&);

Tensor permute(const Tensor &, std::initializer_list<size_t> axes);

double norm2(const Tensor &);
