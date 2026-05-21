import torch
import unittest
from torch.nn.attention import flex_attention

# Helper function to mimic the semantics of tf.keras.backend.set_floatx
# This allows us to leverage the similar API's pattern of setting global float types.
def set_floatx(dtype_str):
    """
    Sets the default float type for PyTorch tensor creation.
    Mimics tf.keras.backend.set_floatx behavior.
    """
    if dtype_str == 'float16':
        torch.set_default_dtype(torch.float16)
    elif dtype_str == 'float32':
        torch.set_default_dtype(torch.float32)
    elif dtype_str == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f'Unknown floatx type: {dtype_str}')

class TestFlexAttentionRecompile(unittest.TestCase):
    def test_flex_attention_dtype_recompile_limit(self):
        """
        Test that flex_attention does not hit the recompile limit 
        when called repeatedly with consistent dtypes.
        
        Bug Description:
        On v2.9.0, flex_attention with torch.compile hit the recompile_limit
        due to dtype validation checks (query.dtype != key.dtype).
        This test verifies that the function can be called multiple times
        (exceeding the default limit of 8) without failing or degrading.
        """
        
        # Leverage the similar API pattern to set the environment
        set_floatx('float32')

        def attention_fn(query, key, value):
            return flex_attention(query, key, value)

        # Compile the function
        compiled_fn = torch.compile(attention_fn)

        # Run the loop more than the default recompile limit (8)
        # to ensure we trigger the condition described in the bug report.
        for _ in range(10):
            # Create inputs. Creating them inside the loop stresses the
            # object ID checks mentioned in the bug log (___check_obj_id).
            query = torch.randn(2, 4, 8, 8)
            key = torch.randn(2, 4, 8, 8)
            value = torch.randn(2, 4, 8, 8)

            # Explicitly check dtypes match, reflecting the validation logic
            # in the bug report: if query.dtype != key.dtype ...
            self.assertEqual(query.dtype, key.dtype)
            self.assertEqual(query.dtype, value.dtype)

            # Execute the compiled function
            output = compiled_fn(query, key, value)
            
            # Basic assertion to ensure execution succeeded
            self.assertIsNotNone(output)
            self.assertEqual(output.shape, (2, 4, 8, 8))

if __name__ == '__main__':
    unittest.main()