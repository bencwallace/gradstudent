#!/bin/sh

set -e

for FILE in $(find examples src tests -name *.cpp -o -name *.h)
do
    clang-format -i $FILE
done
