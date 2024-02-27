#ifndef UTILS_H
#define UTILS_H

// #include <cstddef>
// #include <initializer_list>
#include <memory>

class Array {

private:
  const size_t size;
  std::unique_ptr<size_t[]> data;

public:
  Array(const Array &);
  Array(size_t size);
  Array(std::initializer_list<size_t>);

  size_t operator[](size_t) const;
  size_t &operator[](size_t);
};

#endif
