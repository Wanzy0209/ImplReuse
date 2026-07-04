# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
"""
Test FloorDiv with the specific expression that caused issues:
FloorDiv((24*s37 + 672)*(((s14*s46)//2016)) + 21, 22)
"""

# Create symbolic variables
s14 = sympy.Symbol('s14', integer=True, positive=True)
s37 = sympy.Symbol('s37', integer=True, positive=True) 
s46 = sympy.Symbol('s46', integer=True, positive=True)

print("Testing FloorDiv with complex symbolic expression...")

# Build the numerator expression step by step
inner_expr = FloorDiv(s14 , 2016)  # This creates a FloorDiv
middle_expr = (24 * s37 + 672) * inner_expr
numerator = middle_expr + 21
denominator = 22

print(f"Numerator: {numerator}")
print(f"Denominator: {denominator}")

# Create the FloorDiv expression
result = FloorDiv(numerator, denominator)
print(f"FloorDiv result: {result}")
print(f"FloorDiv result: {sympy.srepr(result)}")