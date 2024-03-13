#pragma once

#include "array.h"

Array defaultStrides(const Array &shape);

size_t sumProd(const Array &, const Array &);

size_t sumProd(const Array &, const Array &, size_t start, size_t end);
