import time
import numpy as np
from functools import wraps
import matplotlib.pyplot as plt

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture start time
        _ = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Capture end time
        # print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return end_time - start_time
    return wrapper


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # current value
        self.avg = 0  # average value
        self.sum = 0  # sum of all values
        self.count = 0  # number of values

    def update(self, val, n=1):
        """Update the meter with new value and count."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def current_stats(self):
        """Prints the current statistics of the meter."""
        print(f"Average: {self.avg*1000:.4f} ms, Count: {self.count}")


def generate_length_list(max_value):
    """Generate a list of powers of two not exceeding max_value and from 2**3."""
    n = np.arange(int(np.log2(max_value)) + 1)[3:]  # Compute all n such that 2^n <= max_value
    powers = 2 ** n  # Vectorized computation of 2^n
    return powers.tolist()  # Convert array to list if needed


def make_length_scaling_plot(df, label):
    # Plotting
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 8))



    # Plot each column on a separate subplot
    axes[0].plot(df['seq_len'], df['n_serial'], marker='x', color='red')
    axes[0].set_title('n_serial')
    axes[0].set_ylabel('count')

    axes[1].plot(df['seq_len'], df['serial_latency'], marker='x', color='blue')
    axes[1].set_title('serial_latency')
    axes[1].set_ylabel('ms')

    axes[2].plot(df['seq_len'], df['batch_latency'], marker='x', color='green')
    axes[2].set_title('batch_latency')
    axes[2].set_ylabel('ms')

    len_xticks = df.seq_len.tolist()
    axes[2].set_xticks(len_xticks)  # Set specific ticks
    axes[2].set_xticklabels(len_xticks)  # Format tick labels

    # Enable grid only along X-axis where ticks are set
    for ax in axes:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    fig.suptitle(label, fontsize=16, fontweight='bold')
    plt.savefig(f'{label}.png')