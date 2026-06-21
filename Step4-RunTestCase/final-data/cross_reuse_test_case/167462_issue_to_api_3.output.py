import torch
import time
import tensorflow as tf

def test_ctc_beam_search_performance():
    """
    Test case to evaluate the performance of tf.compat.v1.nn.ctc_beam_search_decoder
    in a manner analogous to the torch.topk regression reported in Issue 167462.
    
    The original issue highlights a speed regression when selecting top-k elements
    from a large vocabulary (8M) with a large batch size (128) and high k (1000).
    
    This test adapts those parameters to the TensorFlow CTC Beam Search Decoder,
    which performs a similar operation (selecting top-k paths/beam_width) over
    a large set of classes (vocab).
    """
    # Parameters from the original torch.topk regression issue
    BS, VOCAB, K = 128, 8000000, 1000
    
    # CTC decoder expects 3D input: [max_time, batch_size, num_classes]
    # We set max_time=1 to approximate the 2D tensor structure of the original bug
    MAX_TIME = 1

    # Check for GPU availability to match the original "cuda" context
    gpus = tf.config.list_physical_devices('GPU')
    device = "/GPU:0" if gpus else "/CPU:0"
    
    print(f"Running on device: {device}")
    print(f"TensorFlow version: {tf.__version__}")

    with tf.device(device):
        # Initialize inputs
        # Note: Using float32 as ctc_beam_search_decoder typically expects float32 logits.
        # The original issue used float16, but float32 is used here for broader compatibility
        # with the specific TF op implementation.
        logits = tf.random.normal((MAX_TIME, BS, VOCAB), dtype=tf.float32)
        sequence_length = tf.constant([MAX_TIME] * BS, dtype=tf.int32)

        # Warmup runs to allow for any initialization (e.g., kernel loading)
        for _ in range(10):
            _ = tf.compat.v1.nn.ctc_beam_search_decoder(
                logits, 
                sequence_length, 
                beam_width=K
            )

        # Benchmark runs
        walltime = []
        iterations = 100
        
        for _ in range(iterations):
            start = time.time()
            
            # Execute the similar API
            # This operation involves finding top-k paths, analogous to torch.topk
            decoded, _ = tf.compat.v1.nn.ctc_beam_search_decoder(
                logits, 
                sequence_length, 
                beam_width=K
            )
            
            # In TF 2.x eager mode, execution is synchronous. 
            # We capture the wall clock time similar to the original script.
            end = time.time()
            walltime.append(end - start)

        # Calculate average latency, discarding first 10 runs (warmup)
        walltime = walltime[10:]
        avg_latency_ms = 1000 * (sum(walltime) / len(walltime))
        
        print(f"ctc_beam_search_decoder latency: {avg_latency_ms:.4f}ms")
        print(f"Configuration: BS={BS}, VOCAB={VOCAB}, K={K}")

        # Basic assertion to ensure the op ran successfully and returned results
        assert len(decoded) > 0, "Decoder returned no results"
        assert decoded[0].dense_shape.numpy()[1] == BS, "Batch size mismatch in output"

if __name__ == "__main__":
    test_ctc_beam_search_performance()