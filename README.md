# gradstudent

A gradient learner (well, hopefully some day).

`gradstudent` is a learning project. The goal is to build a simple library for manipulating tensors (multidimensional arrays)
and computing the derivatives of these manipulations via (reverse-mode) autograd. Emphasis is placed on accomplishing these
goals with modern C++. Performance alone is not the main objective.

## Applications

See the [examples](examples/README.md) directory.

## Features

Presently, `gradstudent` implements a `Tensor` [class](include/tensor.h) that acts as a container for (strided) multidimensional arrays.
Several simple [operations](include/ops.h) are supported:

* [arithmetic](src/ops/arithmetic.cpp);
* [linear algebra](src/ops/linalg.cpp);
* [views](src/ops/views.cpp);
* [convolution](src/ops/conv.cpp);
* [activations](src/ops/activations.cpp);
* [reductions](src/ops/reductions.cpp).

`gradstudent` also contains the following utilities:

* [PGM image reader/writer](src/utils/image.cpp);
* [NumPy format reader](src/utils/numpy.cpp).

### Future work

The next main hurdle:

* automatic differentiation.

Other things that could be interesting to explore:

* improvements to kernels/optimizations;
* lazy evaluation;
* support for different data types.

Enhancements that would be desirable for the development process:

* Test coverage: gcov/lcov.

## Development

### Requirements

An example [Dockerfile](./Dockerfile) containing the requirements used by this project is given. For the convenience
of VSCode users, a [devcontainer](./.devcontainer.json) configuration is given as well.

Briefly, the requirements are as follows:

* Build: CMake, clang (recommended);
* Documentation: Doxygen;
* Testing: [GoogleTest](https://github.com/google/googletest);
* Benchmarking (limited for now): [Google Benchmark](https://github.com/google/benchmark);
* Code quality: clang-format, clang-tidy, cppcheck.

### Build and test

```
git clone https://github.com/bencwallace/gradstudent.git
cd gradstudent
cmake -B build
cmake --build build -j
cmake --build --target test
```

### Linting and documentation

Git hooks can be found in the `./tools/git` directory. From the repository root, they can be installed as follows:

```
git config core.hooksPath tools/git
```

Scripts for linting and generating documentation can be found in the `./tools` directory. They can be used as follows:

```
./tools/format.sh     # run formatter
./tools/makedocs.sh   # build docs
./tools/lint.sh       # run linter
```
