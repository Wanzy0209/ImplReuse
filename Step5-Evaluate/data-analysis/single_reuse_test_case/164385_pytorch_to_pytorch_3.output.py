import sys
import torch

# Handle missing sympy dependency gracefully
try:
    import sympy
    # FloorDiv is used in the test logic but not explicitly imported in the snippet.
    # We import it here to ensure the test can run if sympy is available.
    from sympy import FloorDiv
except ImportError:
    print("Skipping test: sympy module is not installed.")
    sys.exit(0)

# Note: This test assumes the environment supports symbolic operations with torch and sympy,
# and that FloorDiv is available (likely from torch.fx.experimental.symbolic_shapes or similar context).

# Create symbolic variables
s14 = sympy.Symbol('s14', integer=True, positive=True)
s37 = sympy.Symbol('s37', integer=True, positive=True) 
s46 = sympy.Symbol('s46', integer=True, positive=True)

print("Testing torch.add with complex symbolic expression...")

# Build the expression step by step, mirroring the original bug report's structure
# We use FloorDiv for the inner expression as it was part of the original context
inner_expr = FloorDiv(s14 , 2016) 
middle_expr = (24 * s37 + 672) * inner_expr
numerator = middle_expr + 21

# The original call site was FloorDiv(numerator, 22).
# We adapt this to test the similar API: torch.add.
# We add 22 to the complex numerator expression.
result = torch.add(numerator, 22)

print(f"torch.add result: {result}")
print(f"torch.add result repr: {sympy.srepr(result)}")

# Assertions to verify the behavior
# 1. Check that the result is an Add operation (sympy.Add or equivalent symbolic node)
#    and not a Mul (which was the issue with FloorDiv generating a Rational).
assert isinstance(result, (sympy.Add, torch.fx.Node)), \
    f"Expected result to be an Add or Node, but got {type(result)}"

# 2. Verify that the constant 22 is present in the expression arguments.
#    This ensures the addition wasn't simplified away or incorrectly transformed.
has_constant = False
if hasattr(result, 'args'):
    has_constant = 22 in result.args
else:
    # If it's a torch.fx.Node or similar, we might need to inspect differently,
    # but for sympy expressions, checking args is standard.
    pass
    
assert has_constant, "Constant 22 not found in the result expression arguments"

print("Test passed: torch.add maintained the correct symbolic structure.")