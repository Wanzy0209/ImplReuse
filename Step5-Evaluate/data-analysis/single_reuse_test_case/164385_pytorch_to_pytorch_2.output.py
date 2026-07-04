import torch
import sys

"""
Test Sub with the specific expression that caused issues for FloorDiv:
Sub((24*s37 + 672)*(((s14*s46)//2016)) + 21, 22)
"""

try:
    import sympy
except ImportError:
    print("Skipping test: sympy module is not installed.")
    sys.exit(0)

# Create symbolic variables
s14 = sympy.Symbol('s14', integer=True, positive=True)
s37 = sympy.Symbol('s37', integer=True, positive=True) 
s46 = sympy.Symbol('s46', integer=True, positive=True)

print("Testing Sub with complex symbolic expression...")

# Build the expression step by step
# Note: Assuming FloorDiv is available in the environment as per the original context
inner_expr = FloorDiv(s14 , 2016)  # This creates a FloorDiv
middle_expr = (24 * s37 + 672) * inner_expr
left_operand = middle_expr + 21
right_operand = 22

print(f"Left Operand: {left_operand}")
print(f"Right Operand: {right_operand}")

# Create the Sub expression
# Assuming Sub is the symbolic wrapper for torch.sub
result = Sub(left_operand, right_operand)
print(f"Sub result: {result}")
print(f"Sub result: {sympy.srepr(result)}")