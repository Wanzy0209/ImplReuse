import time
import torch

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    x = torch.randn((BS, VOCAB), device="cuda", dtype=torch.float16)
    walltime = []
    for _ in range(100):
        s = time.time()
        _ = x.topk(k=K, dim=-1)
        torch.cuda.synchronize()
        e = time.time()
        walltime.append(e-s)
    walltime = walltime[10:]
    walltime = sum(walltime) / len(walltime)
    print(f"torch: {torch.__version__}, topk latency: {1000 * walltime}ms")

# torch: 2.4.1+cu126, topk latency: 11.30892170800103ms
# torch: 2.7.1+cu126, topk latency: 17.325512568155922ms