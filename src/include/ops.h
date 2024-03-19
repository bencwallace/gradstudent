#pragma once

#include "tensor.h"

namespace gradstudent {

/* OPERATORS */

Tensor operator+(const Tensor &, const Tensor &);

Tensor operator-(const Tensor &);

Tensor operator-(const Tensor &, const Tensor &);

Tensor operator*(const Tensor &, const Tensor &);

bool operator==(const Tensor &, const Tensor &);

/* LINEAR ALGEBRA */

Tensor dot(const Tensor &, const Tensor &);

Tensor norm2(const Tensor &);

/* VIEWS */

Tensor flatten(Tensor &);
const Tensor flatten(const Tensor &);

Tensor permute(Tensor &, std::initializer_list<size_t> axes);
const Tensor permute(const Tensor &, std::initializer_list<size_t> axes);

Tensor slice(Tensor &, const array_t &);
const Tensor slice(const Tensor &, const array_t &);

template <typename T> T broadcast(T &tensor, const array_t &shape);

template <typename S, typename T> std::tuple<S, T> broadcast(S &, T &);

} // namespace gradstudent
