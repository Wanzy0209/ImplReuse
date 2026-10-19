import torch
import tensorflow as tf
from typing import NamedTuple

# Replacement: torch.Tensor -> tf.feature_column.make_parse_example_spec
class MyNamedTuple(NamedTuple):
    first: tf.feature_column.make_parse_example_spec
    second: tf.feature_column.make_parse_example_spec

class MyNamedTupleSubclass(MyNamedTuple):
    pass

# Replacement: torch.Tensor -> tf.feature_column.make_parse_example_spec
def fn(tup: MyNamedTuple) -> tf.feature_column.make_parse_example_spec:
    # Replacement: torch.tensor -> torch.isinf
    # Original logic: extra_info = torch.tensor(4.0)
    # New logic: Create a tensor and check if it is infinite to generate a value
    input_tensor = torch.tensor(4.0)
    extra_info = torch.isinf(input_tensor)
    
    tup.extra_info = extra_info  # Add dynamic attribute
    return tup

def test_named_tuple_dynamic_attribute():
    print("\nTesting NamedTuple with __setattr__ (Eager):")
    extended_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
    setattr_result = fn(extended_tup)
    
    # Assertion for eager mode
    assert hasattr(setattr_result, 'extra_info'), "Eager mode failed to persist attribute"
    print(f"NamedTuple __setattr__ result: {setattr_result.extra_info}")

    print("\nTesting NamedTuple with __setattr__ (Compiled):")
    extended_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
    compiled_fn = torch.compile(fn, backend="eager")
    setattr_result = compiled_fn(extended_tup)
    
    # This assertion checks for the bug: attribute missing in compiled mode
    try:
        assert hasattr(setattr_result, 'extra_info'), "Compiled mode failed to persist attribute"
        print(f"NamedTuple __setattr__ result: {setattr_result.extra_info}")
    except AttributeError as e:
        print(f"Bug reproduced (AttributeError): {e}")
        raise

if __name__ == "__main__":
    test_named_tuple_dynamic_attribute()