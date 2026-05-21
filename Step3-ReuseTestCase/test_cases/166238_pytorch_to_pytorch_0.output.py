import torch
from collections import defaultdict
import pytest

def test_dynamo_defaultdict_regression():
    """
    Regression test for Issue #166238.
    Verifies that torch.compile can handle collections.defaultdict creation
    without raising torch._dynamo.exc.Unsupported.
    """
    def fn(x):
        # This operation caused the graph break in the bug report:
        # "Dynamo does not know how to trace the function `<class 'collections.defaultdict'>`"
        d = defaultdict(list)
        d['key'].append(x.item())
        return len(d['key'])

    compiled_fn = torch.compile(fn)
    input_tensor = torch.tensor(1.0)
    
    # If the bug is present, this call will raise torch._dynamo.exc.Unsupported
    result = compiled_fn(input_tensor)
    
    assert result == 1

if __name__ == "__main__":
    test_dynamo_defaultdict_regression()