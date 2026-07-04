import tensorflow as tf
import numpy as np

class StridedShardBenchmark(tf.test.Benchmark):
    """
    Test case to reproduce and verify the logic for uneven strided sharding
    described in Issue 168134, adapted to TensorFlow using tf.test.Benchmark.
    
    The bug describes an incorrect distribution of a 1D tensor [0, 1, 2, 3, 4]
    across 2 ranks. The recommended fix involves unflattening, sharding, and
    flattening. This test implements that correct logic in TensorFlow.
    """

    def benchmark_strided_shard_uneven(self):
        # Setup: Simulate the device mesh (2,) using MirroredStrategy
        strategy = tf.distribute.MirroredStrategy()
        
        # Global tensor from the bug report
        global_data = [0., 1., 2., 3., 4.]

        def step_fn():
            """
            Replicates the logic: (5, ) -> unflatten -> (2, 3, ) -> S(1) -> flatten
            """
            ctx = tf.distribute.get_replica_context()
            replica_id = ctx.replica_id_in_sync_group
            num_replicas = ctx.num_replicas_in_sync

            # 1. Unflatten
            # Target shape: (num_replicas, ceil(N / num_replicas))
            # N=5, num_replicas=2 -> (2, 3)
            tensor = tf.constant(global_data)
            n = tf.shape(tensor)[0]
            dim0 = num_replicas
            dim1 = tf.cast(tf.math.ceil(tf.cast(n, tf.float32) / tf.cast(dim0, tf.float32)), tf.int32)

            # Pad to fit the new shape (2 * 3 = 6)
            padded = tf.pad(tensor, [[0, dim0 * dim1 - n]])
            reshaped = tf.reshape(padded, (dim0, dim1))

            # 2. Shard on dim 1 (S(1))
            # We need to split dim 1 (size 3) across 2 replicas unevenly.
            # Rank 0 should get 2 elements, Rank 1 should get 1 element.
            base = dim1 // num_replicas
            remainder = dim1 % num_replicas
            
            # Calculate size for this replica
            size = base + tf.cast(replica_id < remainder, tf.int32)
            
            # Calculate start index for this replica
            start = replica_id * base + tf.minimum(replica_id, remainder)

            # Slice the local shard
            local_shard = reshaped[:, start : start + size]

            # 3. Flatten
            local_tensor = tf.reshape(local_shard, [-1])
            return local_tensor

        # Run the distributed step
        distributed_values = strategy.run(step_fn)
        
        # Gather results to verify correctness
        # Expected: Rank 0 -> [0, 1, 3, 4], Rank 1 -> [2, pad(0)]
        # Gathered order corresponds to replica IDs
        gathered = strategy.gather(distributed_values, axis=0)
        
        # Define expected output based on the bug fix description
        # Rank 0: [0, 1, 3, 4]
        # Rank 1: [2, 0]
        expected = tf.constant([0., 1., 3., 4., 2., 0.])

        # Assertions
        with self.test_session():
            self.assertAllEqual(self.evaluate(gathered), self.evaluate(expected))

        # Report benchmark to leverage the tf.test.Benchmark API
        # (Using a dummy wall_time as this is primarily a correctness test)
        self.report_benchmark(
            iters=1,
            wall_time=0.01,
            name="strided_shard_uneven_correctness"
        )

if __name__ == "__main__":
    tf.test.main()