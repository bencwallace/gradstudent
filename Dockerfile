FROM ubuntu:22.04

RUN apt-get -qq update \
    && apt-get install -qqy --no-install-recommends \
       ca-certificates \
       doxygen \
       wget \
       xz-utils \
    && rm -rf /var/lib/apt/lists/*

ARG CLANG_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.6/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz
RUN wget ${CLANG_URL} -O /tmp/clang.tar.xz \
    && tar -xvkf /tmp/clang.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/clang.tar.xz
RUN ln -s /usr/local/bin/clang /usr/bin/clang

RUN apt-get -qq update \
    && apt-get install -qqy --no-install-recommends \
       build-essential \
       cmake \
       git \
       make \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v1.8.3 https://github.com/google/benchmark.git /tmp/benchmark \
    && mkdir /tmp/benchmark/build \
    && cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release -S /tmp/benchmark -B /tmp/benchmark/build \
    && cmake --build /tmp/benchmark/build --config Release \
    && cmake --build /tmp/benchmark/build --config Release --target install \
    && rm -rf /tmp/benchmark

RUN useradd -m -s /bin/bash ubuntu
