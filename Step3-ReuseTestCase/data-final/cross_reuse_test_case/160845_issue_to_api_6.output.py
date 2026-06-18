import torch
import tensorflow as tf
import numpy as np

def test_dynamic_ragged_shape_complex_consistency():
    """
    Test case for tf.experimental.DynamicRaggedShape inspired by 
    PyTorch Issue 160845 (index_add inconsistency on complex tensors).
    
    This test verifies that DynamicRaggedShape correctly handles the structure
    of complex tensors, ensuring no information loss (similar to the imaginary 
    part loss in the original bug) when defining shapes based on indices.
    """
    
    # Setup: Mimic the original bug's data generation (Real=0, Imag=Random)
    # Original: src_imag = torch.randn(...), src_real = torch.zeros_like(...)
    num_elements = 400
    dtype = tf.complex64
    
    # Generate random imaginary parts and zero real parts
    imag_vals = tf.random.normal((num_elements,), dtype=tf.float32)
    real_vals = tf.zeros_like(imag_vals)
    complex_vals = tf.complex(real_vals, imag_vals)
    
    # Setup: Mimic the original bug's indexing logic
    # Original: idx = torch.arange(400, dtype=torch.long)
    # Here we use row_splits to define the ragged structure based on indices
    # Let's create 4 partitions of 100 elements each
    split_points = [0, 100, 200, 300, 400]
    row_splits = tf.constant(split_points, dtype=tf.int64)
    
    # Operation: Create a RaggedTensor to utilize DynamicRaggedShape
    # This parallels the tensor creation and manipulation in the original bug
    rt = tf.RaggedTensor.from_row_splits(values=complex_vals, row_splits=row_splits)
    
    # Target: Extract the DynamicRaggedShape
    shape = tf.experimental.DynamicRaggedShape.from_tensor(rt)
    
    # Assertions: Verify consistency (Mimicking the CPU vs MPS comparison)
    # 1. Check inner_shape (shape of the flat values)
    # Original bug check: t_cpu.imag.abs().sum() vs t_mps...
    # Here we check if the shape dimension matches the input data length
    assert shape.inner_shape.as_list() == [num_elements], \
        f"Inner shape mismatch. Expected [{num_elements}], got {shape.inner_shape.as_list()}"
        
    # 2. Check row_partitions (structural integrity)
    # Ensure the partitions (indices) are correctly encoded
    assert len(shape.row_partitions) == 1, "Expected 1 row partition"
    
    partition = shape.row_partitions[0]
    assert partition.nrows() == 4, f"Expected 4 rows, got {partition.nrows()}"
    assert partition.nvals() == num_elements, f"Expected {num_elements} values, got {partition.nvals()}"
    
    # 3. Verify the tensor data itself is preserved (checking for the "imaginary loss" equivalent)
    # Ensure the complex values are actually present in the tensor associated with the shape
    imag_sum = tf.math.reduce_sum(tf.math.abs(tf.math.imag(rt.flat_values)))
    assert imag_sum > 0, "Imaginary part sum is zero; data might have been lost or zeroed."
    
    print("Test Passed: DynamicRaggedShape maintains consistency for complex tensors.")

if __name__ == "__main__":
    test_dynamic_ragged_shape_complex_consistency()