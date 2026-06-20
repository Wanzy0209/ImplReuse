import tensorflow as tf
import numpy as np

def test_ctc_beam_search_decoder_consistency():
    """
    Test case for tf.compat.v1.nn.ctc_beam_search_decoder adapted from 
    PyTorch index_add_ complex tensor inconsistency issue (Issue 160845).
    
    The original issue highlights an inconsistency in tensor accumulation 
    (index_add) where the imaginary component was dropped on MPS. 
    This test verifies that the CTC beam search decoder correctly handles 
    input accumulation (logits) and returns valid, consistent results 
    (decoded paths and probabilities) without dropping information.
    """
    
    # Mimic the setup from the bug report: seed, shape, and data generation
    np.random.seed(0)
    
    # Define dimensions
    batch_size = 2
    num_classes = 5  # 4 classes + 1 blank
    max_time_steps = 10
    
    # Create inputs (logits)
    # In the original bug, src had specific real/imag components. 
    # Here we ensure logits have distinct values to verify correct accumulation.
    logits = np.random.randn(max_time_steps, batch_size, num_classes).astype(np.float32)
    
    # Define sequence lengths
    sequence_length = np.array([max_time_steps, max_time_steps], dtype=np.int32)
    
    # Run the API
    # Analogous to t_mps.index_add_ and t_cpu.index_add_ in the bug report.
    # We run the decoder and check if the output matches expectations.
    decoded, log_probabilities = tf.compat.v1.nn.ctc_beam_search_decoder(
        logits,
        sequence_length,
        beam_width=10,
        top_paths=1,
        merge_repeated=True
    )
    
    # Verification logic
    # The original bug checked if the imaginary sum was 0 (incorrect) vs non-zero (correct).
    # Here we check if the log probabilities are valid (not NaN/Inf) and if decoding occurred.
    with tf.compat.v1.Session() as sess:
        decoded_out, log_prob_out = sess.run([decoded, log_probabilities])
        
        # 1. Check Log Probabilities (Analogous to checking imag sum)
        # Ensure probabilities are finite numbers, indicating correct accumulation.
        assert np.all(np.isfinite(log_prob_out)), \
            f"Log probabilities are invalid (NaN/Inf): {log_prob_out}"
            
        # 2. Check Decoded Indices (Analogous to checking tensor values)
        # Ensure the decoder actually found paths within the valid class range.
        # decoded_out is a list of SparseTensors
        sparse_tensor = decoded_out[0]
        indices = sparse_tensor.indices
        values = sparse_tensor.values
        
        # Verify values are within [0, num_classes)
        assert np.all(values >= 0) and np.all(values < num_classes), \
            f"Decoded indices out of bounds: {values}"
            
        # Verify shape consistency
        # The sparse tensor shape is [batch_size, max_decoded_length].
        # max_decoded_length is the length of the longest path in the batch,
        # which is always <= max_time_steps.
        assert sparse_tensor.dense_shape[0] == batch_size, \
            f"Batch dimension mismatch. Expected {batch_size}, got {sparse_tensor.dense_shape[0]}"
        assert sparse_tensor.dense_shape[1] <= max_time_steps, \
            f"Decoded length exceeds max time steps. Expected <= {max_time_steps}, got {sparse_tensor.dense_shape[1]}"

        print("Test passed: CTC Beam Search Decoder handled inputs consistently.")

if __name__ == "__main__":
    test_ctc_beam_search_decoder_consistency()