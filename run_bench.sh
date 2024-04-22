#!/usr/bin/env bash

sleep 10
CUDA_VISIBLE_DEVICES=0 python bench_softmax.py --label bert-large --maxlen 16384 --device cuda --datatype fp16 2>&1 | tee log.bert-large-bs1-maxlen16384-fp16-rtx3090

sleep 60
CUDA_VISIBLE_DEVICES=0 python bench_softmax.py --label bert-large --maxlen 16384 --device cuda --datatype bf16 2>&1 | tee log.bert-large-bs1-maxlen16384-bf16-rtx3090

sleep 10
CUDA_VISIBLE_DEVICES=0 python bench_softmax.py --label bert-large --maxlen 8192 --device cuda --datatype fp32 2>&1 | tee log.bert-large-bs1-maxlen16384-fp32-rtx3090
