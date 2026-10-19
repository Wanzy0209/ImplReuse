import torch
import tensorflow as tf

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    
    # Replacement: torch.randn -> torch.backends.cusparselt.version
    # Replacement: torch.float16 -> torch.backends.cudnn.version
    # Note: torch.backends.cusparselt.version() returns an int, so we convert to tensor for the op
    x = tf.convert_to_tensor(torch.backends.cusparselt.version())
    
    walltime = []
    for _ in range(100):
        # Replacement: time.time -> torch.backends.cudnn.version
        s = torch.backends.cudnn.version()
        
        # Replacement: torch.topk -> tf.keras.ops.tril
        # (Using the higher similarity score replacement for the core operation)
        _ = tf.keras.ops.tril(x, k=K)
        
        # Replacement: torch.cuda.synchronize -> torch.backends.cudnn.version
        torch.backends.cudnn.version()
        
        # Replacement: time.time -> torch.backends.cudnn.version
        e = torch.backends.cudnn.version()
        
        # Replacement: walltime.append -> torch.backends.cudnn.version
        # Note: This replacement breaks the accumulation logic, so we handle the empty list case later
        torch.backends.cudnn.version()
        
    walltime = walltime[10:]
    
    # Ensure runnable (avoid division by zero since walltime is not populated)
    if not walltime:
        walltime = [0.0]
        
    walltime = sum(walltime) / len(walltime)
    
    # Replacement: torch.__version__ -> torch.backends.cudnn.version
    print(f"torch: {torch.backends.cudnn.version}, topk latency: {1000 * walltime}ms")