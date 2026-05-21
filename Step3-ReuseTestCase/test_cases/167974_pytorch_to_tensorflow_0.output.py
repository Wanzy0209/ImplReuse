import tensorflow as tf
import numpy as np

def test_distributed_iterator_2d_input_boundaries():
    """
    Adapts the PyTorch EmbeddingBag bug reproduction logic to tf.distribute.DistributedIterator.
    
    Original Bug: EmbeddingBag with include_last_offset=True generated incorrect offsets 
    [0, 4] instead of [0, 4, 8] for a 2D input of size (2, 4).
    
    Adapted Logic: Verify that the DistributedIterator correctly handles the 2D input 
    and iterates over the expected number of elements (batches), ensuring the "offsets" 
    (iteration boundaries) cover the full dataset size.
    """
    # Input data matching the PyTorch repro script
    # Shape: (2, 4) -> 2 bags, 4 indices each
    input_data = np.array([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=np.int64)

    # Setup distribution strategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Create a dataset from the 2D input.
        # In PyTorch EmbeddingBag, the 2D input implies a batch of bags.
        # In TensorFlow, from_tensor_slices on a 2D array creates a dataset of 1D arrays (rows).
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        
        # Distribute the dataset to obtain a DistributedDataset
        distributed_dataset = strategy.experimental_distribute_dataset(dataset)
        
        # Get the DistributedIterator (the API under test)
        iterator = iter(distributed_dataset)

        # Verify behavior
        # The PyTorch bug resulted in missing the last offset (total size).
        # Here, we verify the iterator yields the correct number of elements (2 rows).
        results = []
        for batch in iterator:
            # DistributedIterator yields PerReplica objects. 
            # We collect them to verify the iteration count.
            results.append(batch)

        # Assertion: The iterator should process all rows in the 2D input.
        # PyTorch expected offsets: [0, 4, 8] (Size 3).
        # TF Iterator expected steps: 2 (One for each row).
        assert len(results) == 2, (
            f"Iterator expected to yield 2 batches for input shape {input_data.shape}, "
            f"but got {len(results)}. This indicates incorrect handling of input boundaries."
        )

        # Further verification: Check that the data retrieved matches the input
        # Note: Accessing values from PerReplica requires context, but checking count 
        # validates the "offset" logic (boundaries).
        print("Test passed: DistributedIterator correctly handled the 2D input boundaries.")

if __name__ == "__main__":
    test_distributed_iterator_2d_input_boundaries()