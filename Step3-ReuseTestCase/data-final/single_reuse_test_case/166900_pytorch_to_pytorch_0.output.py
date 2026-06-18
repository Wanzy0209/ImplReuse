import torch
import torch.utils._pytree as pytree

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Register Bar as a pytree constant, which is part of the trigger for the bug
pytree.register_constant(Bar)

@torch.compile(backend="eager")
def fn(x, obj):
    # The bug occurs when Dynamo tries to generate guards for the temporary variable
    # created during this dictionary assignment involving a pytree constant.
    obj.attr = {3: Bar()}
    return x + 1

if __name__ == "__main__":
    input_tensor = torch.ones(3)
    obj_instance = Foo()
    
    # Execute the compiled function. 
    # In the buggy version, this raises:
    # RuntimeError: Attempt to generate guard on Dynamo-generated temporary variable
    try:
        result = fn(input_tensor, obj_instance)
        
        # Verify the computation result
        expected = input_tensor + 1
        torch.testing.assert_close(result, expected)
        
        # Verify the side effect
        assert isinstance(obj_instance.attr, dict)
        assert 3 in obj_instance.attr
        assert isinstance(obj_instance.attr[3], Bar)
        
        print("Test passed.")
    except RuntimeError as e:
        if "Attempt to generate guard on Dynamo-generated temporary variable" in str(e):
            print(f"Bug reproduced: {e}")
        else:
            raise