# gradstudent

A gradient learner.

`gradstudent` is a learning project. The goal is to build a simple library for manipulating tensors (multidimensional arrays)
and computing the derivatives of these manipulations via (reverse-mode) autograd. Emphasis is placed on accomplishing these
goals with modern C++. Performance alone is not the main objective.

## Features

Presently, `gradstudent` implements a `Tensor` [class](src/include/tensor.h) that acts as a container for (strided) multidimensional arrays.
Several simple [operations](src/include/ops.h) on `Tensor` objects are supported:

* [arithmetic](src/ops/arithmetic.cpp)
* [linear algebra](src/ops/linalg.cpp) (just dot product for now)
* [views](src/ops/views.cpp), including *broadcasting* and a copy-on-write mechanism for views of `const Tensor`s.

The next hurdle is the implementation of automatic differentiation.

Some things that could be interesting to explore:

* parallelization of the operations mentioned above
* lazy evaluation
* support for different backends (e.g. OpenBLAS, CUDA)

## Project structure and code quality

This project uses [CMake](CMakeLists.txt) and compiled with clang (using ASan). GoogleTest, [clang-format](scripts/format.sh),
and [clang-tidy](.clang-tidy) are employed as well.

An example [Docker image](./Dockerfile) containing the requirements used by this project is given. For the convenience
of VSCode users, a [devcontainer](./devcontainer.json) configuration is given.

Additionally, the following would be nice to have: Test coverage (gcov/lcov), CI/CD (GitHub workflows), simple benchmarking scripts.
