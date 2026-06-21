import torch
import time
import tensorflow as tf

def test_ctc_beam_search_decoder_performance():
    """
    Test case for tf.nn.ctc_beam_search_decoder performance.
    This test is derived from a PyTorch topk speed regression issue (Issue ID: 167462).
    It adapts the logic to benchmark the TensorFlow equivalent operation involving
    top-k selection (beam search) on large input dimensions.
    """
    # Check for GPU availability
    if not tf.test.is_gpu_available():
        print("This test requires a GPU to run meaningfully. Skipping.")
        return

    # Parameters from the original torch.topk issue
    # BS: Batch Size
    # VOCAB: Vocabulary size (mapped to num_classes in CTC)
    # K: Top K elements (mapped to beam_width in CTC)
    BS, VOCAB, K = 128, 8000000, 1000
    
    # CTC Beam Search Decoder requires a 3D tensor: [max_time, batch_size, num_classes]
    # We set max_time=1 to approximate the single-step top-k operation of the original test
    # while maintaining the large vocabulary size stress test.
    max_time = 1
    
    print(f"Initializing test with BS={BS}, VOCAB={VOCAB}, K={K}")
    print(f"Input shape: ({max_time}, {BS}, {VOCAB})")
    
    # Create input tensor on GPU
    # Using float16 to match the original test case's dtype
    with tf.device('/GPU:0'):
        logits = tf.random.uniform((max_time, BS, VOCAB), dtype=tf.float16)
        sequence_length = tf.constant([max_time] * BS, dtype=tf.int32)

    # Warmup run to allow for any initialization overhead
    try:
        _ = tf.nn.ctc_beam_search_decoder(logits, sequence_length, beam_width=K)
    except ResourceExhaustedError:
        print("Warmup failed due to OOM. The vocabulary size might be too large for the current GPU.")
        return
    except Exception as e:
        print(f"Warmup failed with error: {e}")
        return

    walltime = []
    iterations = 100
    
    for _ in range(iterations):
        start_time = time.time()
        
        # Execute the similar API: CTC Beam Search Decoder
        # This operation internally performs top-k selection over the vocabulary
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, sequence_length, beam_width=K)
        
        # TensorFlow eager execution is synchronous, so no explicit synchronize needed
        end_time = time.time()
        walltime.append(end_time - start_time)

    # Discard first 10 iterations (warmup) to match original logic
    walltime = walltime[10:]
    
    if not walltime:
        print("Not enough samples collected.")
        return

    avg_latency = sum(walltime) / len(walltime)
    
    print(f"tensorflow: {tf.__version__}, ctc_beam_search_decoder latency: {1000 * avg_latency}ms")
    
    # Basic assertion to ensure the operation completed and returned results
    assert decoded is not None, "Decoder returned None"
    assert len(decoded) > 0, "Decoder returned empty results"

if __name__ == "__main__":
    test_ctc_beam_search_decoder_performance()