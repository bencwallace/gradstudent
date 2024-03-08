#include "tensor.h"

Tensor dot(const Tensor &left, const Tensor &right);

Tensor flatten();

Tensor permute(const Tensor &tensor, std::initializer_list<size_t> axes);

double norm2(const Tensor &);
