# gradstudent

A gradient learner.

`gradstudent` is a learning project. The goal is to build a simple library for manipulating tensors (multidimensional arrays)
and computing the derivatives of these manipulations via (reverse-mode) autograd. Emphasis is placed on accomplishing these
goals with modern C++. Performance alone is not the main objective.

## Applications

See the [examples](examples/README.md) directory.

## Requirements

This project uses [CMake](CMakeLists.txt). clang is the assumed compiler. [Doxygen](Doxyfile) is used to build documentation.
Tools used to help ensure code quality and consistency include ASan, GoogleTest, [clang-format](scripts/format.sh), [cppcheck](scripts/lint.sh), and [clang-tidy](.clang-tidy).

An example [Dockerfile](./Dockerfile) containing the requirements used by this project is given. For the convenience
of VSCode users, a [devcontainer](./.devcontainer.json) configuration is given as well.

In the future, the following would be nice to have: Test coverage (gcov/lcov), CI/CD (GitHub workflows), simple benchmarking scripts.

## Features

Presently, `gradstudent` implements a `Tensor` [class](src/include/tensor.h) that acts as a container for (strided) multidimensional arrays.
Several simple [operations](src/include/ops.h) on `Tensor` objects are supported:

* [arithmetic](src/ops/arithmetic.cpp)
* [linear algebra](src/ops/linalg.cpp) (just dot product for now)
* [views](src/ops/views.cpp), including *broadcasting* and a copy-on-write mechanism for views of `const Tensor`s
* [convolution](src/ops/conv.cpp)

The next hurdle is the implementation of automatic differentiation.

Other things that could be interesting to explore:

* parallelization of the operations mentioned above
* SIMD
* lazy evaluation
* support for different backends (e.g. OpenBLAS, CUDA)
* templatization (support for different tensor data types)

## Build and test

```
git clone https://github.com/bencwallace/gradstudent.git
cd gradstudent
mkdir build
cd build
cmake ..
make -j
ctest -j8
```

**Linting and documentation**

Git hooks can be found in the `./hooks` directory. From the repository root, they can be installed as follows:

```
git config core.hooksPath hooks
```

Scripts for linting and generating documentation can be found in the `./scripts` directory. They can be used as follows:

```
./scripts/format.sh     # run formatter
./scripts/makedocs.sh   # build docs
./scripts/tidy.sh       # run linter
```
