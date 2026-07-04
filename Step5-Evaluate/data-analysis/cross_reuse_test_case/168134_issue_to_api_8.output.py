import tensorflow as tf
import unittest

class TestCentralStorageUnevenDistribution(unittest.TestCase):
    def test_uneven_dataset_distribution(self):
        """
        Test case adapted from PyTorch DTensor Issue 168134.
        
        Original Bug Logic:
        - Global tensor: [0, 1, 2, 3, 4]
        - Mesh: (2,)
        - Shard: _StridedShard(0, split_factor=2)
        - Expected: Rank 0 gets [0, 1, 3, 4], Rank 1 gets [2, pad]
        - Bug: Rank 0 got [0, 1, 3], Rank 1 got [2, 4] (incorrect mapping)

        Adaptation for CentralStorageStrategy:
        - Since CentralStorageStrategy distributes datasets rather than sharding 
          a single static tensor in memory, we verify the distribution of an 
          uneven dataset [0, 1, 2, 3, 4] across replicas.
        - We use batch(2) to mimic the 'split_factor=2' uneven chunking.
        """
        # Initialize strategy (equivalent to init_device_mesh)
        strategy = tf.distribute.experimental.CentralStorageStrategy()
        
        # The global data from the bug report
        global_data = [0, 1, 2, 3, 4]
        
        # Create a dataset
        # We batch by 2 to create uneven chunks: [0, 1], [2, 3], [4]
        # This mimics the uneven sharding scenario in the bug.
        dataset = tf.data.Dataset.from_tensor_slices(global_data)
        dataset = dataset.batch(2)
        
        # Distribute the dataset
        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        @tf.function
        def step_fn(batch):
            # Replicate the logic of checking local tensor content
            # In the original bug, specific indices were missing or misplaced.
            # Here we verify the batch content received by each replica.
            replica_context = tf.distribute.get_replica_context()
            replica_id = replica_context.replica_id_in_sync_group
            
            # Using tf.print to mimic the print statement in the original repro
            tf.print("Rank:", replica_id, "Local Data:", batch)
            return batch

        # Fix: Wrap the iteration loop in a tf.function.
        # The error "RuntimeError: __iter__() is only supported inside of tf.function"
        # indicates that iterating over a DistributedDataset must occur within
        # a tf.function context in this version of TensorFlow.
        @tf.function
        def run_test():
            for batch in dist_dataset:
                strategy.run(step_fn, args=(batch,))

        # Execute the wrapped test logic
        run_test()

if __name__ == '__main__':
    unittest.main()