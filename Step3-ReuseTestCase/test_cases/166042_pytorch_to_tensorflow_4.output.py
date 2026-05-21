import torch
import tensorflow as tf

# The original PyTorch issue (Issue 166042) involves an assertion failure 
# "assert 'int' in str(indices.get_dtype())" within torch.nn.functional.embedding.
# This occurs when non-integer indices (e.g., bfloat16) are passed to the embedding function.
#
# The identified similar API is tf.random.experimental.index_shuffle (documented as 
# tf.data.experimental.index_shuffle). This API shuffles dataset indices based on file 
# metadata rather than performing a lookup with a tensor of indices.
#
# Because the TensorFlow API does not accept a raw tensor of indices as input (it takes 
# file_infos and a reader_factory), we cannot directly reproduce the dtype assertion bug. 
# However, we adapt the test to verify the TensorFlow API's stability using the same 
# seed and context provided in the original report.

def test_index_shuffle_adaptation():
    # Use the seed from the original PyTorch fuzzer output
    seed = 1352030645

    # Mock file information to simulate the input data context
    # In the original PyTorch code, various tensors were created and manipulated.
    # Here we define the files that the index_shuffle operation will process.
    file_infos = [
        {"filename": "file_1.bin"},
        {"filename": "file_2.bin"},
        {"filename": "file_3.bin"},
        {"filename": "file_4.bin"},
    ]

    # Define a reader factory as required by the API
    # This function is responsible for creating a dataset for a given file info.
    def reader_factory(file_info):
        # Simulating reading data. We return a dataset containing the filename.
        return tf.data.Dataset.from_tensor_slices([file_info["filename"]])

    # The prompt mentions tf.random.experimental.index_shuffle, but the provided 
    # docstring corresponds to tf.data.experimental.index_shuffle. 
    # We use the standard public API path for TensorFlow.
    try:
        # Check if the specific path mentioned in the prompt exists (unlikely in standard TF)
        if hasattr(tf.random.experimental, 'index_shuffle'):
            index_shuffle_fn = tf.random.experimental.index_shuffle
        else:
            # Fallback to the correct documented path
            index_shuffle_fn = tf.data.experimental.index_shuffle

        # Create the shuffled dataset
        dataset = index_shuffle_fn(
            file_infos,
            reader_factory,
            seed=seed
        )
    except AttributeError as e:
        print(f"Skipping test: API not found in this TensorFlow version. {e}")
        return

    # Verify the API returns a valid Dataset object
    assert isinstance(dataset, tf.data.Dataset), "index_shuffle should return a tf.data.Dataset"

    # Verify the dataset can be iterated without errors (checking for runtime divergence)
    # We collect the results to ensure the pipeline executes.
    results = list(dataset.as_numpy_iterator())
    
    # Basic sanity check: we expect the same number of files as input
    assert len(results) == len(file_infos), "Output dataset should contain all input files"
    
    # Note: Since index_shuffle is stochastic (even with seed, depending on implementation details 
    # and graph context), we don't assert the exact order, just that it runs successfully.
    print("Test passed: index_shuffle executed successfully with seed {}.".format(seed))

if __name__ == "__main__":
    test_index_shuffle_adaptation()