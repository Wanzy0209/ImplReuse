import torch
import math

# Leveraging the similar API pattern (tf.keras.initializers.HeUniform)
# to generate the input tensor for the test case.
class HeUniform:
    """
    Mimics the behavior of tf.keras.initializers.HeUniform.
    Draws samples from a uniform distribution within [-limit, limit],
    where limit = sqrt(6 / fan_in).
    """
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, dtype=None):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        # Calculate fan_in for the limit
        # Assuming shape is (rows, cols), fan_in is cols
        fan_in = shape[1] if len(shape) > 1 else shape[0]
        limit = math.sqrt(6.0 / fan_in)
        
        return torch.empty(shape, dtype=dtype).uniform_(-limit, limit)

# Original bug reproduction logic
def foo(x):
    c = torch.tensor(7, dtype=torch.uint8)
    return c+x, torch.neg(c), torch.neg(c)+x

def test_neg_add_uint_inductor():
    # Initialize input using the HeUniform-like pattern
    initializer = HeUniform(seed=0)
    x = initializer((2, 2), dtype=torch.float32)
    
    print(f"Input x: {x}")

    # Compile the function
    cfoo = torch.compile(foo)
    
    # Execute eager and compiled
    res = foo(x)
    cres = cfoo(x)

    # Check results
    print(f"res[0] (c+x): {res[0]}")
    print(f"cres[0] (c+x): {cres[0]}")
    assert torch.allclose(res[0], cres[0]), "Mismatch in c+x"

    print(f"res[1] (neg(c)): {res[1]}")
    print(f"cres[1] (neg(c)): {cres[1]}")
    assert torch.equal(res[1], cres[1]), "Mismatch in neg(c)"

    # The bug manifests here: neg(c) should be 249 (uint8 wrap), not -7
    print(f"res[2] (neg(c)+x): {res[2]}")
    print(f"cres[2] (neg(c)+x): {cres[2]}")
    
    # This assertion will fail if the bug is present
    assert torch.allclose(res[2], cres[2]), \
        f"Mismatch in neg(c)+x. Expected {res[2]}, got {cres[2]}"

if __name__ == "__main__":
    test_neg_add_uint_inductor()