# gradstudent

A gradient learner.

`gradstudent` is a learning project. The goal is to build a simple library for manipulating tensors (multidimensional arrays)
and computing the derivatives of these manipulations via (reverse-mode) autograd. Emphasis is placed on accomplishing these
goals with modern C++. Performance alone is not the main objective.

## Applications

See the [examples](examples/README.md) directory.

## Requirements

An example [Dockerfile](./Dockerfile) containing the requirements used by this project is given. For the convenience
of VSCode users, a [devcontainer](./.devcontainer.json) configuration is given as well.

This project uses [CMake](CMakeLists.txt). [Doxygen](Doxyfile) is used to build documentation.
Tools used to help ensure code quality include ASan, [clang-format](tools/format.sh), [cppcheck](tools/lint.sh), and [clang-tidy](.clang-tidy).
For correctness and performance, [GoogleTest](https://github.com/google/googletest) and [Google Benchmark](https://github.com/google/benchmark) are used.

In the future, the following would be nice to have: Test coverage (gcov/lcov), CI/CD (GitHub workflows).

## Features

Presently, `gradstudent` implements a `Tensor` [class](include/tensor.h) that acts as a container for (strided) multidimensional arrays.
Several simple [operations](include/ops.h) on `Tensor` objects are supported:

* [arithmetic](src/ops/arithmetic.cpp);
* [linear algebra](src/ops/linalg.cpp);
* [views](src/ops/views.cpp);
* [convolution](src/ops/conv.cpp).

The next hurdle is the implementation of automatic differentiation.

Other things that could be interesting to explore:

* optimizations (SIMD, multithreading, cache-friendly implementations);
* lazy evaluation;
* support for different backends (e.g. OpenBLAS, CUDA);
* templatization (support for different data types).

## Build and test

```
git clone https://github.com/bencwallace/gradstudent.git
cd gradstudent
cmake -B build
cmake --build build -j
cmake --build --target test
```

**Linting and documentation**

Git hooks can be found in the `./tools/git` directory. From the repository root, they can be installed as follows:

```
git config core.hooksPath tools/git
```

Scripts for linting and generating documentation can be found in the `./scripts` directory. They can be used as follows:

```
./tools/format.sh     # run formatter
./tools/makedocs.sh   # build docs
./tools/lint.sh       # run linter
```
