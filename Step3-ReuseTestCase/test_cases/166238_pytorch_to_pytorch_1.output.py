import torch
import collections
import functools

def test_reduce_with_defaultdict():
    """
    Test case adapted from the bug report regarding collections.defaultdict creation
    and the similar API functools.reduce (using the torch._dynamo.polyfills implementation).
    """
    def fn(x):
        # Original bug context: creating a defaultdict inside torch.compile
        # This was causing: torch._dynamo.exc.Unsupported: Unsupported function call
        d = collections.defaultdict(list)
        d['a'].append(x)
        
        # Similar API: using functools.reduce
        # This verifies if the reduce polyfill handles the defaultdict correctly
        result = functools.reduce(lambda acc, val: acc + val, d['a'], 0)
        return result

    # Compile the function
    compiled_fn = torch.compile(fn)
    
    # Run the test with a tensor input
    input_val = torch.tensor(5)
    output = compiled_fn(input_val)
    
    # Verify the result matches the expected behavior
    assert output == 5

if __name__ == "__main__":
    test_reduce_with_defaultdict()