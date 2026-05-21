import torch
import tensorflow as tf

def test_to_variant_serialization_multiple_runs():
    """
    Test case for tf.data.experimental.to_variant based on the logic of 
    PyTorch AOT precompile serialization bug (Issue 165447).
    
    The original bug involves serializing a compiled model, resetting state,
    loading it back, and verifying correctness, specifically checking for
    failures when running multiple times.
    
    This test adapts that logic to TensorFlow by:
    1. Creating and processing a Dataset (analogous to defining and compiling a Module).
    2. Serializing it to a variant tensor (analogous to save_compiled_function).
    3. Reconstructing the dataset from the variant (analogous to load_compiled_function).
    4. Verifying the output matches the expected result.
    5. Running the process multiple times to ensure stability.
    """

    # Define a simple dataset transformation (Analogous to class M)
    def create_dataset():
        return tf.data.Dataset.range(10).map(lambda x: x * 2).batch(4)

    # Expected output for the dataset
    expected_output = [[0, 2, 4, 6], [8, 10, 12, 14], [16, 18]]

    # Run the serialization/deserialization cycle multiple times
    # to address the "running multiple times" aspect of the bug title.
    for _ in range(2):
        # 1. Create and process the dataset
        dataset = create_dataset()

        # 2. Serialize the dataset to a variant tensor
        # This is the API under test, analogous to save_compiled_function
        variant_tensor = tf.data.experimental.to_variant(dataset)

        # Verify the variant tensor was created successfully
        assert variant_tensor is not None
        assert variant_tensor.dtype == tf.variant
        assert variant_tensor.shape == tf.TensorShape([])

        # 3. Deserialize the dataset from the variant tensor
        # This is analogous to load_compiled_function, necessary to verify integrity
        reconstructed_dataset = tf.data.experimental.from_variant(
            variant_tensor, 
            dataset.element_spec
        )

        # 4. Verify execution (Analogous to assert torch.allclose)
        results = list(reconstructed_dataset.as_numpy_iterator())
        
        assert len(results) == len(expected_output), \
            f"Output length mismatch: {len(results)} vs {len(expected_output)}"
        
        for res, exp in zip(results, expected_output):
            assert (res == exp).all(), f"Output mismatch: {res} vs {exp}"

if __name__ == "__main__":
    test_to_variant_serialization_multiple_runs()
    print("Test passed.")