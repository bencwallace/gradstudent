#!/bin/bash

set -e

mkdir -p data

BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES=(
#   "train-images-idx3-ubyte.gz"
#   "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

for file in "${FILES[@]}"
do
  wget -P data $BASE_URL$file
  gunzip -f data/$file
#   rm data/$file
done
