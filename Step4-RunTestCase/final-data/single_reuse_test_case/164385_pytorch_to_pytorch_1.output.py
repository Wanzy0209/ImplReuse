import sys
import torch
"""
Test Mul with the specific expression related to FloorDiv issues:
Mul((24*s37 + 672)*(((s14*s46)//2016)) + 21, 22)
"""

try:
    import sympy
    from sympy import Mul, Symbol, Integer
except ImportError:
    print("Skipping test: sympy module is not installed.")
    sys.exit(0)

# Create symbolic variables
s14 = Symbol('s14', integer=True, positive=True)
s37 = Symbol('s37', integer=True, positive=True) 
s46 = Symbol('s46', integer=True, positive=True)

print("Testing Mul with complex symbolic expression...")

# Build the numerator expression step by step
# Using // for FloorDiv as per standard SymPy usage in this context
inner_expr = s14 // 2016
middle_expr = (24 * s37 + 672) * inner_expr
numerator = middle_expr + 21
denominator = 22

print(f"Numerator: {numerator}")
print(f"Denominator: {denominator}")

# Create the Mul expression (corresponding to torch.mul)
result = Mul(numerator, denominator)
print(f"Mul result: {result}")
print(f"Mul result: {sympy.srepr(result)}")
print(f"Result type: {type(result)}")

# Assertion to verify the result is a Mul instance
assert isinstance(result, Mul), f"Expected Mul, got {type(result)}"