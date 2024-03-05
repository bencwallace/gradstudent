#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <memory>

class Array {

private:
  std::unique_ptr<size_t[]> data;

public:
  const size_t size;

  Array(const Array &);
  Array(size_t size);
  Array(std::initializer_list<size_t>);

  size_t operator[](size_t) const;
  size_t &operator[](size_t);
  bool operator!=(const Array &) const;

  friend std::ostream &operator<<(std::ostream &, Array const &);
  friend Array zerosArray(size_t);
};

std::ostream &operator<<(std::ostream &, Array const &);
Array zerosArray(size_t);

#endif
