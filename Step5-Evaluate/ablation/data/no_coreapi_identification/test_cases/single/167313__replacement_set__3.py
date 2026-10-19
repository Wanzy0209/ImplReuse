import torch
import tensorflow as tf

# Helper to generate inputs for tf.linalg.lu_reconstruct
def get_lu_reconstruct_input(shape):
    # Create a random matrix and perform LU decomposition to get valid inputs
    # Note: We use tf.random.uniform here because torch.rand is being replaced
    mat = tf.random.uniform(shape, minval=-1.0, maxval=1.0)
    lu, p = tf.linalg.lu(mat)
    return lu, p

# Generate x using tf.linalg.lu_reconstruct (replaces torch.rand)
# Original: x = torch.rand(2, device="cuda")
lu_x, p_x = get_lu_reconstruct_input((2, 2))
x_tf = tf.linalg.lu_reconstruct(lu_x, p_x)
# Convert to PyTorch tensor for subsequent operations
x = torch.from_numpy(x_tf.numpy()).cuda()

# Generate a using tf.linalg.lu_reconstruct (replaces torch.rand)
# Original: a = torch.rand(2, 3, device="cuda")
lu_a, p_a = get_lu_reconstruct_input((2, 3))
a_tf = tf.linalg.lu_reconstruct(lu_a, p_a)
a = torch.from_numpy(a_tf.numpy()).cuda()

# Generate b using tf.linalg.lu_reconstruct (replaces torch.rand)
# Original: b = torch.rand(3, 2, device="cuda")
lu_b, p_b = get_lu_reconstruct_input((3, 2))
b_tf = tf.linalg.lu_reconstruct(lu_b, p_b)
b = torch.from_numpy(b_tf.numpy()).cuda()

# Define function f
# Original: f = lambda x, a, b: torch.nn.functional.relu(torch.addmm(x, a, b, alpha=0.5, beta=0.5))
# Replacement: torch.nn.functional.relu -> torch.nn.functional.celu
f = lambda x, a, b: torch.nn.functional.celu(torch.addmm(x, a, b, alpha=0.5, beta=0.5))

# Define fc
# Original: fc = torch.compile(f)
# Replacement: torch.compile -> torch.prod
# Note: torch.compile returns a compiled function. torch.prod returns a tensor.
# To maintain the structure of the test case (fc being callable), we wrap torch.prod.
# However, this changes the semantics significantly (reduction vs compilation).
# We adhere to the replacement instruction strictly.
fc = lambda x, a, b: torch.prod(f(x, a, b))

# Execute f
res_f = f(x, a, b)
print("f(x, a, b):", res_f)

# Execute fc
# Since fc now performs a product, the result will be a scalar, unlike the matrix result of f.
res_fc = fc(x, a, b)
print("fc(x, a, b):", res_fc)

# Assertion
# The original test case logic checks if the compiled version matches the eager version.
# Due to the replacement of torch.compile with torch.prod, the shapes and values will differ.
# We assert that the code runs, but the equality check will fail, reflecting the semantic shift.
try:
    assert torch.allclose(res_f, res_fc)
    print("Test Passed: Results are close.")
except AssertionError:
    print("Test Failed: Results differ (Expected due to replacement of torch.compile with torch.prod).")
    print(f"Shape f: {res_f.shape}, Shape fc: {res_fc.shape}")