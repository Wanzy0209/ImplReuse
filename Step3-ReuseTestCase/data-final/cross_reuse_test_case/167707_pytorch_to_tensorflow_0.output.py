import torch
import tensorflow as tf
import gzip
import io

def test_tf_compat_as_text_trace_handling():
    """
    Adapted test case for Issue 167707.
    
    Original Bug: torch.profiler.profile with tensorboard_trace_handler 
    (use_gzip=True) saves a broken trace.
    
    Adaptation: Verify that tf.compat.as_text correctly handles the conversion
    of raw bytes (simulating a read from a gzipped trace file) back to 
    valid text, ensuring the trace data is not "broken".
    """
    
    # Simulate the trace data content, including the worker_name from the original bug
    worker_name = "trace"
    trace_data = f'{{"worker": "{worker_name}", "events": [{{"name": "cuda_op"}}]}}'

    # Simulate the 'use_gzip=True' behavior: compressing the data to bytes
    # This mimics what torch.profiler.tensorboard_trace_handler does internally
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(trace_data.encode('utf-8'))
    
    # Simulate reading the file back (which yields raw bytes)
    buf.seek(0)
    with gzip.GzipFile(fileobj=buf, mode='rb') as f:
        raw_bytes = f.read()

    # Use the similar API: tf.compat.as_text
    # This corresponds to the internal logic of reading the trace file
    # and ensuring it is valid text (not broken)
    try:
        decoded_text = tf.compat.as_text(raw_bytes)
    except Exception as e:
        raise AssertionError(f"Failed to decode trace data using tf.compat.as_text: {e}")

    # Verify the trace is not broken (content matches original)
    assert decoded_text == trace_data, "Decoded trace does not match original, data is broken"
    
    # Verify pass-through behavior for already text data (robustness check)
    assert tf.compat.as_text(trace_data) == trace_data

    print("Test passed: Trace data integrity verified via tf.compat.as_text.")

if __name__ == "__main__":
    test_tf_compat_as_text_trace_handling()