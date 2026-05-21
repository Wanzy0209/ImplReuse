import torch
import tensorflow as tf

# Adapted test case for tf.strings.substr based on the structure of the torch.clamp bug report.
# Note: The original bug involves numerical clamping on the MPS backend. 
# This test adapts the structural logic (instantiation, operation, verification) 
# to the TensorFlow string operation API.

# Original: a = torch.zeros(1, device='mps')
# Adapted: Create a string tensor
a = tf.constant(["hello"])

# Original: a_clamped = a.clamp(min=0.0) (Trigger line)
# Adapted: Substr with length 0 (Edge case)
a_sub = tf.strings.substr(a, pos=0, len=0)
print("Substr len=0:", a_sub)

# Original: b = torch.zeros(1, device='mps'); c = b.clamp(min=1e-7)
# Adapted: Substr with length 1 (Minimum length)
b = tf.constant(["hello"])
print("Original:", b)
c = tf.strings.substr(b, pos=0, len=1)
print("Substr len=1:", c)
assert c.numpy()[0] == b"h", "Expected 'h'"

# Original: b = torch.zeros(1, device='mps'); c = b.clamp(min=1e-7, max=None)
# Adapted: Substr with explicit unit parameter
b = tf.constant(["hello"])
print("Original:", b)
c = tf.strings.substr(b, pos=0, len=1, unit="BYTE")
print("Substr len=1, unit=BYTE:", c)
assert c.numpy()[0] == b"h", "Expected 'h'"

# Original: b = torch.zeros(1, device='mps'); c = b.clamp(min=1e-7, max=torch.inf)
# Adapted: Substr with large length (effectively no max limit)
b = tf.constant(["hello"])
print("Original:", b)
c = tf.strings.substr(b, pos=0, len=100)
print("Substr len=100:", c)
assert c.numpy()[0] == b"hello", "Expected 'hello'"

# Original: b = torch.zeros(1, device='mps'); c = b.clamp_min(1e-7)
# Adapted: Substr with length 1 (Similar to min constraint)
b = tf.constant(["hello"])
print("Original:", b)
c = tf.strings.substr(b, pos=0, len=1)
print("Substr len=1:", c)
assert c.numpy()[0] == b"h", "Expected 'h'"