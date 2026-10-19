import tensorflow as tf
import unittest

class TestFloatHandlingWithSummaryValue(unittest.TestCase):
    def test_rebind_unbacked_float_handling(self):
        """
        Test case adapted from the PyTorch issue #162480.
        The original bug involved missing float handling in rebind_unbacked().
        This test preserves the logic of checking for float types and discarding them,
        but replaces the logging mechanism (log.info) with tf.compat.v1.Summary.Value
        as requested.
        """
        
        # Simulating the inputs that would be processed in the loop
        # Format: (n.target, raw_u0, path, u1)
        test_data = [
            ("node_1", "val_1", "path_a", 10),       # Integer
            ("node_2", "val_2", "path_b", 3.14),     # Float (Should be discarded)
            ("node_3", "val_3", "path_c", "string"), # String
            ("node_4", "val_4", "path_d", 2.71),     # Float (Should be discarded)
        ]

        processed_values = []
        discarded_summaries = []

        for n_target, raw_u0, path, u1 in test_data:
            # Original logic: Check if u1 is a float
            if isinstance(u1, float):
                # Replacement: log.info -> tf.compat.v1.Summary.Value
                # We create a Summary.Value to record the discarded float
                summary_val = tf.compat.v1.Summary.Value(
                    tag=f"rebind_unbacked_discard_{n_target}",
                    simple_value=u1
                )
                discarded_summaries.append(summary_val)
                
                # Original logic: continue to skip processing
                continue

            # Process non-float values
            processed_values.append(u1)

        # Assertions to verify the logic
        # 1. Ensure floats were skipped
        self.assertEqual(len(processed_values), 2)
        self.assertIn(10, processed_values)
        self.assertIn("string", processed_values)
        self.assertNotIn(3.14, processed_values)
        self.assertNotIn(2.71, processed_values)

        # 2. Ensure tf.compat.v1.Summary.Value was created for floats
        self.assertEqual(len(discarded_summaries), 2)
        
        # Verify the content of the summaries
        self.assertEqual(discarded_summaries[0].tag, "rebind_unbacked_discard_node_2")
        self.assertEqual(discarded_summaries[0].simple_value, 3.14)
        
        self.assertEqual(discarded_summaries[1].tag, "rebind_unbacked_discard_node_4")
        self.assertEqual(discarded_summaries[1].simple_value, 2.71)

if __name__ == "__main__":
    unittest.main()