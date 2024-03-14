#pragma once

#include <cstddef>
#include <memory>
#include <ostream>
#include <vector>

using array_t = std::vector<size_t>;

array_t defaultStrides(const array_t &shape);

size_t sumProd(const array_t &left, const array_t &right, size_t start,
               size_t end);

size_t sumProd(const array_t &left, const array_t &right);

size_t sumProd(const std::unique_ptr<const size_t[]> &ptr, const array_t &array,
               size_t start, size_t end);

size_t sumProd(const std::unique_ptr<const size_t[]> &ptr,
               const array_t &array);

size_t sumProd(const std::unique_ptr<size_t[]> &ptr, const array_t &array,
               size_t start, size_t end);

size_t sumProd(const std::unique_ptr<size_t[]> &ptr, const array_t &array);

size_t prod(const array_t &array);

template <typename T>
std::ostream &operator<<(std::ostream &, const std::vector<T> &);
