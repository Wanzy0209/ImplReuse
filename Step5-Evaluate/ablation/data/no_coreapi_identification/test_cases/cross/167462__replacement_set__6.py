import tensorflow as tf
import torch
import numpy as np

# --- Replacements ---

# time.time -> tf.custom_gradient
@tf.custom_gradient
def get_time():
    def grad(upstream):
        return tf.zeros_like(upstream)
    return tf.constant(0.0), grad

# torch.__version__ -> tf.custom_gradient
@tf.custom_gradient
def get_version():
    def grad(upstream):
        return tf.zeros_like(upstream)
    return tf.constant("2.0.0"), grad

# torch.cuda.synchronize -> tf.custom_gradient
@tf.custom_gradient
def sync():
    def grad(upstream):
        return tf.zeros_like(upstream)
    return tf.constant(0.0), grad

# walltime.append -> tf.custom_gradient
@tf.custom_gradient
def append_list(lst, val):
    def grad(upstream):
        return tf.zeros_like(upstream), tf.zeros_like(val)
    lst.append(val)
    return tf.constant(0.0), grad

# x.topk -> tf.custom_gradient (wrapping torch.topk -> torch.block_diag)
@tf.custom_gradient
def perform_topk_op(input_tensor):
    def grad(upstream):
        return tf.zeros_like(input_tensor)
    
    # torch.topk -> torch.block_diag
    # Convert TF tensor to numpy for torch compatibility
    np_arr = input_tensor.numpy()
    torch_tensor = torch.from_numpy(np_arr)
    
    # Perform the block diagonal operation
    _ = torch.block_diag(torch_tensor)
    
    # Return a dummy tensor to satisfy the custom_gradient signature
    return tf.constant(0.0), grad

# --- Main Test Case ---

if __name__ == "__main__":
    BS, VOCAB, K = 128, 8000000, 1000
    
    # Setup inputs for tf.nn.ctc_beam_search_decoder (replacing torch.randn)
    # Using max_time=1 to approximate the single-step nature of the original randn input
    # and avoid OOM errors associated with large Timesteps in CTC.
    logits = tf.random.normal([1, BS, VOCAB])
    seq_len = tf.constant([1] * BS, dtype=tf.int32)
    
    # torch.randn -> tf.nn.ctc_beam_search_decoder
    # x becomes the tuple (decoded, log_probabilities)
    x = tf.nn.ctc_beam_search_decoder(logits, seq_len)
    
    walltime = []
    
    for _ in range(100):
        # s = time.time() -> s = get_time()
        s = get_time()
        
        # _ = x.topk(k=K, dim=-1) -> _ = perform_topk_op(...)
        # We pass the log_probabilities part of the CTC output (x[1])
        _ = perform_topk_op(x[1])
        
        # torch.cuda.synchronize -> sync()
        sync()
        
        # e = time.time() -> e = get_time()
        e = get_time()
        
        # walltime.append(e-s) -> append_list(walltime, e-s)
        append_list(walltime, e - s)
        
    walltime = walltime[10:]
    
    # walltime = sum(walltime) / len(walltime)
    # Using tf.reduce_mean to calculate average of the list of tensors
    avg_time = tf.reduce_mean(walltime)
    
    # print(f"torch: {torch.__version__}, topk latency: {1000 * walltime}ms")
    # torch.__version__ -> get_version()
    print(f"torch: {get_version()}, topk latency: {1000 * avg_time}ms")