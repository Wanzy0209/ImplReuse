import torch
import tensorflow as tf

def test_get_strategy_in_graph_mode():
    """
    Test case for tf.distribute.get_strategy inside a tf.function (graph mode).
    This mirrors the original bug report where torch._check caused issues inside
    torch.compile. Here we verify that get_strategy behaves correctly within
    a compiled graph context.
    """
    # Setup a distributed strategy to provide a context
    strategy = tf.distribute.MirroredStrategy()

    # The original bug used @torch.compile(fullgraph=True).
    # In TensorFlow, the equivalent is @tf.function with jit_compile=True.
    @tf.function(jit_compile=True)
    def check_strategy_context(x):
        # Leverage the similar API: tf.distribute.get_strategy
        # This mimics the call to torch._check in the original bug,
        # testing if accessing the strategy context breaks the graph.
        current_strategy = tf.distribute.get_strategy()

        # Mimic the "check" logic from the original bug.
        # The original code checked a condition (x.shape[0] > 3).
        # Here we perform a check on the strategy to ensure it is valid.
        # We use tf.debugging.assert to stay within graph ops.
        tf.debugging.assert_not_equal(
            current_strategy, 
            None, 
            message="Strategy should not be None inside graph"
        )
        
        return x + 1

    # Execute the test within the strategy scope
    with strategy.scope():
        # Create a tensor similar to the repro (x = torch.randn(3, ...))
        x = tf.constant([1.0, 2.0, 3.0, 4.0])
        
        try:
            result = check_strategy_context(x)
            print("Test passed. Result:", result)
            assert tf.reduce_all(result == x + 1)
        except Exception as e:
            print(f"Test failed with error: {e}")
            raise

if __name__ == "__main__":
    test_get_strategy_in_graph_mode()