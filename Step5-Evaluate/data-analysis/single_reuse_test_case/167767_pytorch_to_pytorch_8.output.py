import torch
import sys

# Check if MPS backend is available before running the tests
if not torch.backends.mps.is_available():
    print("MPS backend is not available. Skipping test.")
    sys.exit(0)

# Adapted test case for torch.tril based on the torch.clamp MPS bug report.
# The original bug involved incorrect value modification (clamping) on MPS.
# Here we test if torch.tril correctly zeroes out the upper triangle on MPS.

# Setup: Create a 2D tensor of ones to verify the zeroing behavior
a = torch.ones(3, 3, device='mps')
# Mimic the "trigger" line from the original bug report
a_triled = torch.tril(a)

# Test Case 1: Basic tril
b = torch.ones(3, 3, device='mps')
print("Original b:\n", b)
c = torch.tril(b)
print("Tril b:\n", c)
# Assertion: Upper triangle elements should be 0
assert c[0, 1] == 0.0, "Expected upper triangle element to be 0"

# Test Case 2: tril with diagonal argument (k=1)
b = torch.ones(3, 3, device='mps')
print("Original b:\n", b)
c = torch.tril(b, k=1)
print("Tril b (k=1):\n", c)
# Assertion: First diagonal above main should be 1, second should be 0
assert c[0, 1] == 1.0, "Expected first diagonal above main to be 1"
assert c[0, 2] == 0.0, "Expected second diagonal above main to be 0"

# Test Case 3: tril with diagonal argument (k=-1)
b = torch.ones(3, 3, device='mps')
print("Original b:\n", b)
c = torch.tril(b, k=-1)
print("Tril b (k=-1):\n", c)
# Assertion: Main diagonal should be 0
assert c[0, 0] == 0.0, "Expected main diagonal to be 0"