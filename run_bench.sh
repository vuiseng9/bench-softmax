#!/usr/bin/env bash

sleep 60
MAXLEN=16384
DTYPE=bf16
numactl --cpunodebind=0 --membind=0 python bench_softmax.py --label cache-bert-large-ipex --maxlen 1024 --device cpu --datatype $DTYPE 2>&1 | tee log.bert-large-bs1-maxlen${MAXLEN}-${DTYPE}-spr1

sleep 60
MAXLEN=16384
DTYPE=fp32
numactl --cpunodebind=0 --membind=0 python bench_softmax.py --label cache-bert-large-ipex --maxlen 1024 --device cpu --datatype $DTYPE 2>&1 | tee log.bert-large-bs1-maxlen${MAXLEN}-${DTYPE}-spr1