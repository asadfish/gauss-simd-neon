#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "== build =="
g++ main.cc -O2 -fopenmp -pthread -o main

echo "== quick correctness/performance test =="
./main 256 1 2 4 | tee result_256.csv

echo "== recommended experiment =="
./main 512 1 2 4 8 | tee result_512.csv
./main 1024 1 2 4 8 | tee result_1024.csv

echo "== optional larger case, run only if time is enough =="
echo "./main 1500 1 2 4 8 | tee result_1500.csv"
echo "./main 2048 1 2 4 8 | tee result_2048.csv"
