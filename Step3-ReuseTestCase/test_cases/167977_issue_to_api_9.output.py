import tensorflow as tf
import numpy as np

# The original issue describes a failure in PyTorch DTensor where sharding propagation
# failed because the internal logic did not account for DTensors passed as keyword arguments.
# The similar API is tf.types.experimental.distributed.Mirrored, which represents values
# kept in sync across replicas (typically accessed via tf.distribute.MirroredStrategy).
# This test verifies that TensorFlow's MirroredStrategy correctly handles distributed
# tensors (MirroredValues) when they are passed as keyword arguments to operations.

class TestMirroredTensorKwargs(tf.test.TestCase):

    def test_mirrored_strategy_with_tensor_kwargs(self):
        """
        Tests that distributed tensors (MirroredVariables) can be passed as keyword arguments
        to functions executed within a MirroredStrategy scope.
        
        This mirrors the PyTorch bug scenario where `aten.min.dim_min` received tensors
        via the `out` kwarg, causing an assertion error in strategy propagation.
        """
        # Initialize the MirroredStrategy
        strategy = tf.distribute.MirroredStrategy()

        # Define a function that accepts tensors via keyword arguments.
        # This simulates the custom operation signature from the PyTorch issue.
        def compute_with_kwargs(x, *, tensor_kwarg_1, tensor_kwarg_2):
            # Perform a computation using the positional and kwarg tensors
            return x + tensor_kwarg_1 + tensor_kwarg_2

        with strategy.scope():
            # Create distributed variables (DTensor equivalents)
            # x: [[1, 2], [3, 4]]
            var_x = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
            
            # tensor_kwarg_1: [[5, 5], [5, 5]]
            var_kwarg_1 = tf.Variable([[5.0, 5.0], [5.0, 5.0]])
            
            # tensor_kwarg_2: [[10, 10], [10, 10]]
            var_kwarg_2 = tf.Variable([[10.0, 10.0], [10.0, 10.0]])

            # Execute the function passing distributed tensors as kwargs.
            # In the PyTorch bug, this caused an AssertionError because the strategy
            # expansion logic only checked 'args' and ignored 'kwargs'.
            # Here we verify that the TensorFlow equivalent handles this correctly.
            result = strategy.run(
                compute_with_kwargs,
                args=(var_x,),
                kwargs={
                    'tensor_kwarg_1': var_kwarg_1,
                    'tensor_kwarg_2': var_kwarg_2
                }
            )

            # Expected result: [[1+5+10, 2+5+10], [3+5+10, 4+5+10]] = [[16, 17], [18, 19]]
            expected = np.array([[16.0, 17.0], [18.0, 19.0]])

            # strategy.run returns a PerReplica object. We verify the result on the local replica.
            self.assertAllClose(
                expected,
                strategy.experimental_local_results(result)[0].numpy()
            )

if __name__ == "__main__":
    tf.test.main()