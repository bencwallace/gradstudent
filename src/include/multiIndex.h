#pragma once

#include "array.h"

class MultiIndex : public Array {

private:
  const Array shape;

  void increment(size_t);
  void reset();

public:
  MultiIndex(const Array &);

  MultiIndex &operator=(const Array &);
  MultiIndex operator++();
};
