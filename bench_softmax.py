import argparse
import torch
import time
import pandas as pd

import time
from datetime import datetime
from functools import partial
from tqdm import tqdm
from collections import OrderedDict
from utils import AverageMeter, timeit, generate_length_list, make_length_scaling_plot


DEVICE_LIST=['cpu', 'cuda']
DATATYPE_MAP={
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

BREAKTIME = 5
pause = partial(time.sleep, BREAKTIME)
softmax_fn = partial(torch.nn.functional.softmax, dim=-1)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking softmax.")
    
    parser.add_argument('--nwarmup', type=int, default=10,
                        help='Number of warm-up cycles to run before the actual benchmark. Default is 10.')

    parser.add_argument('--nloop', type=int, default=100,
                        help='Number of benchmark cycles to run. Default is 100.')
    
    parser.add_argument('--batchsize', type=int, default=1,
                        help='Number of benchmark cycles to run. Default is 1.')
    
    parser.add_argument('--maxlen', type=int, default=128,
                        help='softmax input length limit, benchmarking from from 2**3 to 2**(x) < length_limit, cannot be lower than 8')
    
    parser.add_argument('--input_length', metavar='N', type=int, nargs='+',
                    help='an optional list of target length for benchmark, will be combined with maxlen')
    
    parser.add_argument('--nhead', type=int, default=16,
                        help='number of self-attention head. Default is 16 (bert-large).')
    
    parser.add_argument('--device', choices=DEVICE_LIST, default=DEVICE_LIST[-1], 
                        help=f'any of {DEVICE_LIST}')
    
    parser.add_argument('--datatype', choices=list(DATATYPE_MAP.keys()), default='fp16', 
                        help=f'any of {list(DATATYPE_MAP.keys())}')

    parser.add_argument('--label', type=str, default=None,
                        help='Optional label for the process. Defaults to None.')
    
    return parser.parse_args()


@timeit
def bench_serial(n_calls, input_length, device, dtype):
    with torch.no_grad():
        for _ in range(n_calls):
            _ = softmax_fn(torch.randn(input_length, device=device, dtype=dtype))

@timeit
def bench_batch(bs, n_head, seq_len, device, dtype):
    with torch.no_grad():
        _ = softmax_fn(torch.randn(bs, n_head, seq_len, seq_len, device=device, dtype=dtype))


def main():
    args = parse_args()

    n_warmup = args.nwarmup
    n_loop = args.nloop
    n_head = args.nhead
    bs = args.batchsize
    device = args.device
    dtype = DATATYPE_MAP[args.datatype]

    assert args.maxlen >= 8, "--maxlen must be larger than or equal 8"
    bench_length_list = generate_length_list(args.maxlen)

    if args.input_length is not None:
        bench_length_list.extend(args.input_length)
        bench_length_list = sorted(bench_length_list)

    if args.label is None:
        outlabel = datetime.now().strftime("%y%m%d_%H%M%S")
    else:
        outlabel = args.label

    bench_data = []

    for seq_len in bench_length_list:
        print(f"\n{'-'*100}")
        serial_meter = AverageMeter()
        batch_meter = AverageMeter()

        # serial ---------------------------------------
        n_softmax = bs * n_head * seq_len
        softmax_len = seq_len

        entry_dict = OrderedDict()
        entry_dict['seq_len'] = seq_len
        entry_dict['n_serial'] = n_softmax

        print(f"[len: {seq_len}][Serial]: Warming up ...")
        for _ in range(n_warmup):
            bench_serial(n_softmax, softmax_len, device, dtype)

        for _ in tqdm(range(n_loop), desc=f"[len: {seq_len}][Serial][n_calls: {n_softmax}]: Benchmarking ..."):
            serial_meter.update(bench_serial(n_softmax, softmax_len, device, dtype))
        
        entry_dict['serial_latency'] = serial_meter.avg*1000
        serial_meter.current_stats()
        pause()

        # batch ---------------------------------------
        print(f"[len: {seq_len}][Batch]: Warming up ...")
        for _ in range(n_warmup):
            bench_batch(bs, n_head, softmax_len, device, dtype)

        for _ in tqdm(range(n_loop), desc=f"[len: {seq_len}][Batch][{bs}, {n_head}, {softmax_len}, {softmax_len}]: Benchmarking ..."):
            batch_meter.update(bench_batch(bs, n_head, softmax_len, device, dtype))

        entry_dict['batch_latency'] = batch_meter.avg*1000
        batch_meter.current_stats()
        pause()

        bench_data.append(entry_dict)

    # post-processeing
    df = pd.DataFrame.from_dict(bench_data)

    df['n_warmup'] = n_warmup
    df['n_loop'] = n_loop
    df['n_head'] = n_head
    df['batch_size'] = bs
    df['device'] = device
    df['dtype'] = args.datatype

    df = df[['n_warmup', 'n_loop',  'device', 'dtype', 'n_head', 'batch_size','seq_len', 'n_serial', 'serial_latency', 'batch_latency']]

    outname = f"{outlabel}_{df.device[0]}_{df.dtype[0]}_head={df.n_head[0]}_maxlen={df.seq_len.tolist()[-1]}_nloop={df.n_loop[0]}_warmup={df.n_warmup[0]}"

    df.to_csv(f"{outname}.csv")
    make_length_scaling_plot(df, label=outname)

if __name__ == "__main__":
    main()


