
### Run
```
$ python bench_softmax.py --help
usage: bench_softmax.py [-h] [--nwarmup NWARMUP] [--nloop NLOOP] [--batchsize BATCHSIZE] [--maxlen MAXLEN] [--nhead NHEAD] [--device {cpu,cuda}] [--datatype {fp32,fp16,bf16}]
                        [--label LABEL]

Benchmarking script for performance testing.

options:
  -h, --help            show this help message and exit
  --nwarmup NWARMUP     Number of warm-up cycles to run before the actual benchmark. Default is 10.
  --nloop NLOOP         Number of benchmark cycles to run. Default is 100.
  --batchsize BATCHSIZE
                        Number of benchmark cycles to run. Default is 1.
  --maxlen MAXLEN       softmax input length limit, benchmarking from from 2**3 to 2**(x) < length_limit, cannot be lower than 8
  --nhead NHEAD         number of self-attention head. Default is 16 (bert-large).
  --device {cpu,cuda}   any of ['cpu', 'cuda']
  --datatype {fp32,fp16,bf16}
                        any of ['fp32', 'fp16', 'bf16']
  --label LABEL         Optional label for the process. Defaults to None.
```

### Sample
```
# Single GPU
CUDA_VISIBLE_DEVICES=0 python bench_softmax.py --label bert-large --maxlen 1024 --device cuda --datatype fp16

# CPU
numactl --cpunodebind=0 --membind=0 python bench_softmax.py --label bert-large --maxlen 1024 --device cpu --datatype bf16
```

### Dependency
```
pip install torch pandas matplotlib
```