import torch
import tensorflow as tf
import numpy as np

def test_sum_with_huge_step_slice():
    """
    Test case for tf.keras.metrics.Sum based on the logic of the PyTorch bug report.
    
    Original Bug: torch.slice_copy with a huge step (2**63 - 1) creates an empty tensor.
    Inductor (compiler) crashes on this empty tensor, while eager mode handles it.
    
    Adaptation: We use tf.strided_slice with a huge step to generate an empty tensor,
    then feed it into tf.keras.metrics.Sum inside a tf.function (graph mode/compiled)
    to ensure it handles the edge case gracefully without crashing.
    """
    
    # 1. Setup input tensor (mimicking get_input(n=875))
    x = tf.random.normal((875,), dtype=tf.float32)
    
    # 2. Perform the slice with a huge step (mimicking the bug trigger)
    # PyTorch: torch.slice_copy(x, dim=0, start=449, step=(2**63 - 1))
    # TensorFlow: tf.strided_slice
    huge_step = 2**63 - 1
    start = 449
    
    # This operation should result in an empty tensor
    sliced_tensor = tf.strided_slice(x, [start], [tf.shape(x)[0]], [huge_step])
    
    # Verify eager mode behavior (should be empty)
    assert tf.equal(tf.size(sliced_tensor), 0), "Eager slice should result in empty tensor"
    
    # 3. Initialize the Similar API: tf.keras.metrics.Sum
    metric = tf.keras.metrics.Sum()
    
    # Initialize with some data to have a baseline
    metric.update_state([1.0, 2.0, 3.0])
    baseline_result = metric.result().numpy()
    
    # 4. Define a compiled function (mimicking torch.compile/Inductor)
    @tf.function
    def update_and_compute(metric_obj, data):
        metric_obj.update_state(data)
        return metric_obj.result()

    # 5. Execute the test case
    # The original bug caused a Segmentation Fault here.
    # We assert that TF handles the empty tensor input correctly.
    try:
        result = update_and_compute(metric, sliced_tensor)
        
        # The sum should remain unchanged because we added an empty tensor
        assert result.numpy() == baseline_result, \
            f"Sum should remain {baseline_result} after adding empty tensor, got {result.numpy()}"
            
        print("Test Passed: tf.keras.metrics.Sum handles empty slice (huge step) in graph mode.")
        
    except Exception as e:
        print(f"Test Failed: Exception raised in graph mode: {e}")
        raise

if __name__ == "__main__":
    test_sum_with_huge_step_slice()