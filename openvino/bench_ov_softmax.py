import itertools
import subprocess
from argparse import Namespace
from dataclasses import dataclass
from typing import List
import pandas as pd
import re
import os
import time

model_map={
    "bert-l-attn-softmax-bf16": "ovir_softmax_bf16/softmax.xml",
    "bert-l-attn-softmax-f32": "ovir_softmax_fp32/softmax.xml"
}

length_list = [
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
]

LOGROOT="./ov-softmax-length-scaling"
NITER=100
SLEEP_SEC=30

@dataclass
class BenchmarkResult:
    stdout: str = ''
    stderr: str = ''
    avg_latency: float = 0.
    throughput: float = 0.


# def run_bapp_latency_mode_with_pc_dump(cmd):

        # avg_line = filter(None, stdout.split('\n')[-4].split())
        # throughput_line = filter(None, stdout.split('\n')[-1].split())
        # avg_line = list(avg_line)
        # throughput_line = list(throughput_line)
        # assert 'Average:' in avg_line and 'Throughput:' in throughput_line

        # return BenchmarkResult(
        #     stdout=stdout,
        #     stderr=stderr,
        #     avg_latency=float(list(avg_line)[-2]),
        #     throughput=float(list(throughput_line)[-2]),
        # )

# results = []

for sl in length_list:
    for mlabel, mpath in model_map.items():
        folder = os.path.join(f"{LOGROOT}/sl_{sl:04}_{mlabel}")
        os.makedirs(folder, exist_ok=True)
        dtype=mlabel.split("-")[-1]
        dtype='bf16'
        time.sleep(SLEEP_SEC)
        print(f"\n\nRunning {folder}")
        cmd=f'numactl --cpunodebind 1 --membind 1 benchmark_app -api sync -niter {NITER} -m {mpath} -shape [1,16,{sl},{sl}] -infer_precision {dtype} -report_type average_counters -report_folder {folder}'
        print(f"cmd: \n{cmd}")
        
        with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
            p.wait()
            stdout = p.stdout.read().decode()
            stderr = p.stderr.read().decode()

            if len(stdout) > 0:
                with open(f"{folder}/log.stdout", "w") as f:
                    f.write(stdout)
                
                stdout_lines = stdout.strip().split('\n')
                avg_line = filter(None, stdout_lines[-4].split())
                throughput_line = filter(None, stdout_lines[-1].split())
                avg_line = list(avg_line)
                throughput_line = list(throughput_line)

                avg_latency=float(avg_line[-2])
                throughput=float(throughput_line[-2])
                print(f'--> Avg. latency.: {avg_latency}; Throughput: {throughput}')

            if len(stderr) > 0:
                print("stderr!!")
                with open(f"{folder}/log.stderr", "w") as f:
                    f.write(stderr) 
            
            with open(f"{folder}/cmd.txt", "w") as f:
                f.write(cmd)
