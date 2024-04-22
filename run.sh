#!/usr/bin/env bash

numactl --cpunodebind=0 --membind=0 python bench_softmax.py --label bert-large --maxlen 16384 --device cpu --datatype bf16
