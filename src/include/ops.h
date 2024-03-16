#pragma once

#include "tensor.h"

void checkCompatibleShape(const Tensor &, const Tensor &);

Tensor operator+(const Tensor &, const Tensor &);

Tensor operator-(const Tensor &);

Tensor operator-(const Tensor &, const Tensor &);

Tensor operator*(const Tensor &, const Tensor &);

bool operator==(const Tensor &, const Tensor &);

Tensor dot(const Tensor &, const Tensor &);

Tensor flatten(const Tensor &);

Tensor permute(const Tensor &, std::initializer_list<size_t> axes);

double norm2(const Tensor &);
