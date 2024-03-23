#!/bin/sh

set -e

while getopts "v" opt; do
    case ${opt} in
        v)
            ARGS=--extra-arg=-v
            ;;
    esac
done

cppcheck -i build --project=build/compile_commands.json
find examples src -name *.cpp -o -name *.h -exec clang-tidy -p build ${ARGS} {} \;
