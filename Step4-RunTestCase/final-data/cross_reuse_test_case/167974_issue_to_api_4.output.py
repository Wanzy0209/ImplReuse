import torch
import unittest

# Handle TensorFlow import errors due to environment issues (e.g., libstdc++ version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    tf = None
    TF_IMPORT_ERROR = str(e)

class TestEmbeddingBagOffsets(unittest.TestCase):
    @unittest.skipIf(tf is None, f"Skipping test: TensorFlow not available due to environment error - {TF_IMPORT_ERROR}")
    def test_embeddingbag_offsets_with_tf_dlpack(self):
        """
        Test case for Issue 167974: EmbeddingBag sum has incorrect offsets when input is 2D 
        and include_last_offset is set to True.
        
        This test leverages tf.experimental.dlpack.to_dlpack to generate the input tensor,
        ensuring the bug reproduction logic is preserved while utilizing the similar API
        for data transfer between frameworks.
        """
        
        # 1. Create the input data in TensorFlow.
        # The original bug report uses a 2D tensor of shape (2, 4).
        tf_input = tf.constant([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=tf.int64)

        # 2. Leverage the similar API: tf.experimental.dlpack.to_dlpack.
        # This converts the TensorFlow tensor into a DLPack capsule, allowing for zero-copy
        # transfer to other frameworks.
        dlpack_capsule = tf.experimental.dlpack.to_dlpack(tf_input)

        # 3. Convert the DLPack capsule back to a PyTorch tensor.
        # This allows us to feed the data into torch.nn.EmbeddingBag.
        torch_input = torch.from_dlpack(dlpack_capsule)

        # 4. Initialize the EmbeddingBag layer with the problematic configuration.
        # include_last_offset=True is the flag that triggers the incorrect offset generation.
        embedding_sum = torch.nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)

        # 5. Execute the forward pass.
        # When input is 2D, offsets are auto-generated.
        # Bug: generates [0, 4] (length 2).
        # Expected: [0, 4, 8] (length 3).
        output = embedding_sum(torch_input)

        # 6. Assertions.
        # With include_last_offset=True, the number of bags is calculated as len(offsets) - 1.
        # If offsets are [0, 4] (Bug), num_bags = 1. Output shape would be (1, 3).
        # If offsets are [0, 4, 8] (Correct), num_bags = 2. Output shape should be (2, 3).
        # Since the input has 2 rows, we expect 2 bags.
        expected_shape = (2, 3)
        
        assert output.shape == expected_shape, (
            f"Bug reproduced: Expected output shape {expected_shape} but got {output.shape}. "
            "This indicates that offsets were generated as [0, 4] instead of [0, 4, 8]."
        )

if __name__ == "__main__":
    unittest.main()