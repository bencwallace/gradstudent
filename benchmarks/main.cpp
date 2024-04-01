#include <benchmark/benchmark.h>

#include "gradstudent.h"

using namespace gradstudent;

static void BM_Add(benchmark::State& state) {
  Tensor x({10000, 10000});
  Tensor y({10000, 10000});
  for (auto _ : state)
    x + y;
}
BENCHMARK(BM_Add);

// static void BM_Dot(benchmark::State& state) {
//   Tensor input({1000, 1000});
//   Tensor kernel({3, 3});
//   for (auto _ : state)
//     conv(input, kernel);
// }
// BENCHMARK(BM_Dot);

BENCHMARK_MAIN();
