#include <memory>

#include "array.h"

class TensorData {

private:
  const size_t size_;
  std::shared_ptr<double[]> data;

public:
  TensorData(size_t);
  TensorData(const TensorData &);

  size_t size() const;

  TensorData &operator=(const TensorData &);
  double operator[](size_t) const;
  double &operator[](size_t);

};
