#include <benchmark/benchmark.h>

#include "gradstudent.h"

using namespace gs;

static void BM_Dot(benchmark::State& state) {
  Tensor input({1000, 1000});
  Tensor kernel({3, 3});
  for (auto _ : state)
    conv(input, kernel);
}
BENCHMARK(BM_Dot);

BENCHMARK_MAIN();
