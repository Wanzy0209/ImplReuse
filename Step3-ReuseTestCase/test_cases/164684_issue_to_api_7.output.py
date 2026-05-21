import torch
import unittest

# The issue describes a bug in torch.compile where multiplying a scalar boolean 
# (extracted from a tensor) with a FakeTensor fails.
# The similar API is tf.keras.optimizers.schedules.serialize.
# We generate a test case that mimics the structure of the similar API 
# (serializing a module/schedule) while preserving the bug reproduction logic 
# (scalar * tensor operation).

def serialize_schedule(module_bool, weight):
    """
    Simulates a serialization-like step where a boolean configuration 
    determines the output, similar to how tf.keras.optimizers.schedules.serialize
    processes a module string.
    """
    # Reproduce the bug logic: extract scalar bool and multiply with tensor
    # This triggers: TypeError("unsupported operand type(s) for *: 'SymBool' and 'FakeTensor'")
    flag = module_bool.squeeze().item()
    result = flag * weight
    
    # Preserve logic from original bug report
    if result.is_complex():
        result = result.real
        
    # Return a tuple to match the signature pattern of the similar API
    return result, 1

class TestScheduleSerializationCompile(unittest.TestCase):
    def test_serialize_schedule_with_mul(self):
        # Configuration required to trigger the specific code path in dynamo
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        # Setup inputs
        # arg_0 mimics a configuration flag (bool tensor)
        arg_0 = torch.randint(0, 2, (1,), dtype=torch.bool) > 0
        # sentinel mimics a weight or tensor data requiring gradients
        sentinel = torch.tensor(1.0, requires_grad=True)

        # 1. Test Eager Execution
        try:
            eager_result, eager_version = serialize_schedule(arg_0, sentinel)
            print(' eager success')
        except Exception as e:
            self.fail(f"Eager execution failed unexpectedly: {e}")

        # 2. Test Compiled Execution
        # This is where the bug manifests due to SymBool and FakeTensor interaction
        compiled_program = torch.compile(serialize_schedule, fullgraph=True, dynamic=True)
        
        try:
            compiled_result, compiled_version = compiled_program(arg_0, sentinel)
            print(' compile success')
            
            # Verify correctness if compilation succeeds
            self.assertTrue(torch.allclose(eager_result, compiled_result))
            self.assertEqual(eager_version, compiled_version)
            
        except TypeError as e:
            # Check for the specific error mentioned in the bug report
            if "unsupported operand type(s) for *: 'SymBool' and 'FakeTensor'" in str(e):
                self.fail(f"Bug reproduced: {e}")
            else:
                raise

if __name__ == '__main__':
    unittest.main()