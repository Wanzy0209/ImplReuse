import time
import torch

def test_topk_performance_regression():
    """
    Reproduces the torch.topk speed regression reported in Issue 167462.
    Leverages torch.backends.cudnn.version to correlate performance with the backend version.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    # Leverage the similar API: torch.backends.cudnn.version
    # Checking the backend version is crucial when diagnosing performance regressions
    # in CUDA operations like topk.
    cudnn_version = torch.backends.cudnn.version()
    print(f"cuDNN version: {cudnn_version}")

    # Parameters from the original bug report
    BS, VOCAB, K = 128, 8000000, 1000
    
    # Create input tensor
    x = torch.randn((BS, VOCAB), device="cuda", dtype=torch.float16)

    walltime = []
    
    # Benchmark loop
    for _ in range(100):
        s = time.time()
        _ = x.topk(k=K, dim=-1)
        torch.cuda.synchronize()
        e = time.time()
        walltime.append(e - s)

    # Discard warmup iterations (first 10)
    walltime = walltime[10:]
    
    # Calculate average latency
    avg_walltime = sum(walltime) / len(walltime)
    latency_ms = 1000 * avg_walltime

    print(f"torch: {torch.__version__}, topk latency: {latency_ms:.4f}ms")

    # Assertion to ensure the operation completes and latency is recorded
    assert latency_ms > 0, "Latency measurement failed"

if __name__ == "__main__":
    test_topk_performance_regression()