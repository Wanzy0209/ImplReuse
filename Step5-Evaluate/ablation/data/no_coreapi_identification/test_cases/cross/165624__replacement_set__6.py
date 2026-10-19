import torch
import tensorflow as tf
from unittest.mock import MagicMock, patch

def test_merge_mistake_double_execution_with_replacement():
    """
    Test case to reproduce the logic of Issue 165624 (double execution of a pass)
    while replacing the source API (joint_custom_pre_pass) with tf.keras.ops.relu
    and leveraging the similar API (torch.backends.opt_einsum.get_opt_einsum).
    """
    
    # Mock the configuration and graph objects
    class MockConfig:
        joint_custom_pre_pass = None
        joint_graph_constant_folding = False

    class MockGraph:
        pass

    config = MockConfig()
    graph = MockGraph()

    # Replacement: joint_custom_pre_pass -> tf.keras.ops.relu
    # We assign the replacement API to the config slot used in the original code
    config.joint_custom_pre_pass = tf.keras.ops.relu

    # Track execution count to verify the bug (double execution)
    execution_count = 0

    # Mock GraphTransformObserver to intercept calls
    with patch('torch._inductor.fx_passes.joint_graph.GraphTransformObserver') as MockObserver:
        def mock_apply_graph_pass(self, pass_func):
            nonlocal execution_count
            # Simulate the pass execution using the replacement API (tf.keras.ops.relu)
            # We apply it to a dummy tensor to ensure it runs
            dummy_input = tf.constant([-1.0, 0.0, 1.0])
            _ = pass_func(dummy_input)
            execution_count += 1

        # Setup the mock instance behavior
        mock_instance = MagicMock()
        mock_instance.apply_graph_pass = mock_apply_graph_pass
        mock_instance.apply_gm_pass = lambda self, x: None  # Mock other pass
        MockObserver.return_value = mock_instance

        # Leverage similar API: torch.backends.opt_einsum.get_opt_einsum
        # We use this to check availability, mirroring the conditional logic pattern
        opt_einsum = torch.backends.opt_einsum.get_opt_einsum()
        print(f"Opt_einsum status: {opt_einsum is not None}")

        # --- Original Buggy Code Block (Adapted) ---
        count = 0

        # First invocation block
        if config.joint_custom_pre_pass is not None:
            MockObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
                config.joint_custom_pre_pass
            )
            count += 1

        # Middle section (mocked imports and other passes)
        # from .post_grad import remove_noop_ops
        MockObserver(graph, "remove_noop_ops").apply_graph_pass(lambda x: x)

        if config.joint_graph_constant_folding:
            MockObserver(graph, "constant_fold_uniform_value").apply_gm_pass(
                lambda x: x
            )

        # Second invocation block (The Merge Mistake)
        if config.joint_custom_pre_pass is not None:
            MockObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
                config.joint_custom_pre_pass
            )
            count += 1
        # --- End Original Buggy Code Block ---

        # Assertions
        # We expect 2 because the bug causes the code to run twice
        assert execution_count == 2, f"Expected 2 executions (reproducing bug), got {execution_count}"
        assert count == 2, f"Expected count to be 2, got {count}"

if __name__ == "__main__":
    test_merge_mistake_double_execution_with_replacement()
    print("Test passed successfully.")