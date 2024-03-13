#include <memory>

#include "array.h"

class TensorData {

protected:
  const size_t size_;

public:
  TensorData(size_t);
  TensorData(const TensorData &);

  size_t size() const;

  virtual double operator[](size_t) const = 0;
  virtual double &operator[](size_t) = 0;

  virtual ~TensorData(){};
};

class TensorDataCpu : public TensorData {

private:
  std::unique_ptr<double[]> data;

public:
  TensorDataCpu(size_t);

  double operator[](size_t) const override;
  double &operator[](size_t) override;
};
