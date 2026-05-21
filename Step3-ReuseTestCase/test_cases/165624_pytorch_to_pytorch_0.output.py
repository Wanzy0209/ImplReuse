import torch
import unittest

class TestJointCustomPrePass(unittest.TestCase):
    def test_joint_custom_pre_pass_not_executed_twice(self):
        """
        Regression test for issue #165624.
        Verifies that joint_custom_pre_pass is executed exactly once during compilation,
        not twice due to a merge mistake in joint_graph.py.
        """
        call_count = 0

        def custom_pre_pass(gm):
            """A simple custom pass that increments a counter to track executions."""
            nonlocal call_count
            call_count += 1
            return gm

        def simple_model(x):
            return x + 1

        # Compile the model using the inductor backend and pass the custom pre-pass
        compiled_model = torch.compile(
            simple_model,
            backend="inductor",
            options={"joint_custom_pre_pass": custom_pre_pass}
        )

        # Run the compiled model to trigger the compilation process
        input_tensor = torch.randn(10)
        _ = compiled_model(input_tensor)

        # Assert that the custom pass was called exactly once.
        # If the bug (duplicate code block) is present, call_count will be 2.
        self.assertEqual(call_count, 1, 
                         f"joint_custom_pre_pass was called {call_count} times, expected 1. "
                         "This indicates the duplicate execution bug in joint_graph.py.")

if __name__ == "__main__":
    unittest.main()