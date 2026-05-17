import torch
from typing import NamedTuple, List

# Define the NamedTuple structure
class MyNamedTuple(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor

# Define a subclass to allow dynamic attributes
class MyNamedTupleSubclass(MyNamedTuple):
    pass

def process_features(tuples: List[MyNamedTuple]) -> List[MyNamedTuple]:
    """
    Processes a list of NamedTuples, adding dynamic attributes.
    This function structure mimics the pattern of tf.feature_column.make_parse_example_spec
    where a collection of items is processed.
    """
    results = []
    for tup in tuples:
        extra_info = torch.tensor(4.0)
        # Add dynamic attribute - this is the core of the bug report
        tup.extra_info = extra_info
        results.append(tup)
    return results

def test_named_tuple_dynamic_attributes():
    # Setup inputs
    # Mimicking the feature_columns set/list from the similar API usage
    input_tuples = [
        MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0)),
        MyNamedTupleSubclass(first=torch.tensor([3.0]), second=torch.tensor(2.0))
    ]
    
    # Test 1: Eager execution
    print("Testing Eager execution:")
    eager_results = process_features(input_tuples)
    for res in eager_results:
        assert hasattr(res, 'extra_info'), "Eager mode failed to set attribute"
        assert torch.equal(res.extra_info, torch.tensor(4.0)), "Eager mode attribute value mismatch"
        print(f"Eager result extra_info: {res.extra_info}")

    # Test 2: Compiled execution (eager backend)
    print("\nTesting Compiled execution (eager backend):")
    compiled_fn = torch.compile(process_features, backend="eager")
    compiled_results = compiled_fn(input_tuples)
    
    for res in compiled_results:
        # This assertion will fail if the bug exists
        assert hasattr(res, 'extra_info'), "Compiled mode failed to persist attribute"
        assert torch.equal(res.extra_info, torch.tensor(4.0)), "Compiled mode attribute value mismatch"
        print(f"Compiled result extra_info: {res.extra_info}")

if __name__ == "__main__":
    test_named_tuple_dynamic_attributes()