import torch
import time

torch.manual_seed(0)

# Mapping the original matmul shapes to RNN parameters.
# Original shapes: ((1, 12, 10, 64), (1, 12, 64, 10))
# Interpretation: Batch=1, Seq=12, Hidden=10, Input=64
configs = [
    {"input_size": 64, "hidden_size": 10},
    {"input_size": 10, "hidden_size": 64},
]

def benchmark_rnn(config, dtype=torch.float16, device="cpu", repeat=500):
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    batch_size = 1
    seq_len = 12

    # torch.nn.RNN inherits from torch.nn.RNNBase
    rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    rnn.to(dtype=dtype).to(device)

    # Create input tensor: (Batch, Seq_Len, Input_Size)
    # Shape: (1, 12, input_size)
    X = torch.empty(batch_size, seq_len, input_size, dtype=dtype, device=device).uniform_(0,1) * 2 - 1

    # warm up
    for _ in range(5000):
        _ = rnn(X)

    # run
    times = []
    for i in range(repeat):
        start = time.time()
        _ = rnn(X)
        end = time.time()
        if i > 100:
            times.append(round((end - start) * 1000 * 1000))
    times.sort()
    print(times)
    avg_time_us = sum(times) / len(times)
    return avg_time_us

if __name__ == "__main__":
    for config in configs:
        t = benchmark_rnn(config)
        print(f"RNN (Input={config['input_size']}, Hidden={config['hidden_size']}) -> {t:.3f} us")