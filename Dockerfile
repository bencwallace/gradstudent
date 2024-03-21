FROM ubuntu:22.04

RUN apt-get -qq update \
    && apt-get install -qqy --no-install-recommends \
        bash-completion \
        build-essential \
        cmake \
        ca-certificates \
        doxygen \
        gdb \
        git \
        make \
        openssh-client \
        sudo \
        wget \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

ARG CLANG_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.6/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz
RUN wget ${CLANG_URL} -O /tmp/clang.tar.xz \
    && tar -xvkf /tmp/clang.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/clang.tar.xz
RUN ln -s /usr/local/bin/clang /usr/bin/clang

RUN useradd -m -s /bin/bash -G sudo vscode \
    && passwd -d vscode
