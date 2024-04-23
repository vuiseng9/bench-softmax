#!/usr/bin/env bash

# sleep 10
# MAXLEN=16384
# CUDA_VISIBLE_DEVICES=0 python bench_softmax.py --label bert-large --maxlen $MAXLEN --device cuda --datatype fp16 2>&1 | tee log.bert-large-bs1-maxlen${MAXLEN}-fp16-rtx3090

# sleep 60
# MAXLEN=16384
# CUDA_VISIBLE_DEVICES=0 python bench_softmax.py --label bert-large --maxlen $MAXLEN --device cuda --datatype bf16 2>&1 | tee log.bert-large-bs1-maxlen${MAXLEN}-bf16-rtx3090

# sleep 10
# MAXLEN=8192
# CUDA_VISIBLE_DEVICES=0 python bench_softmax.py --label bert-large --maxlen $MAXLEN --device cuda --datatype fp32 2>&1 | tee log.bert-large-bs1-maxlen${MAXLEN}-fp32-rtx3090

sleep 10
MAXLEN=16384
OTHER_LEN="9216 10240 11264 12288 13312 14336 15360"
CUDA_VISIBLE_DEVICES=0 python bench_softmax.py --label bert-large-custom-len --maxlen $MAXLEN --input_length $OTHER_LEN --device cuda --datatype fp16 2>&1 | tee log.bert-large-bs1-maxlen${MAXLEN}+others-fp16-rtx3090

sleep 60
MAXLEN=16384
OTHER_LEN="9216 10240 11264 12288 13312 14336 15360"
CUDA_VISIBLE_DEVICES=0 python bench_softmax.py --label bert-large-custom-len --maxlen $MAXLEN --input_length $OTHER_LEN --device cuda --datatype bf16 2>&1 | tee log.bert-large-bs1-maxlen${MAXLEN}+others-bf16-rtx3090