import torch
import torch.nn as nn
import torch.optim as optim
import types

# --- Compatibility Setup for PyTorch 1.x vs 2.x ---
HAS_NEW_API = False
try:
    from torch.library import Library, impl, register_fake
    HAS_NEW_API = True
except ImportError:
    # Mocks for PyTorch 1.x where these APIs do not exist
    class Library:
        def __init__(self, ns, kind):
            pass
        def define(self, sig):
            pass
    
    def impl(name, dispatch_key):
        def decorator(func):
            return func
        return decorator
    
    def register_fake(name):
        def decorator(func):
            return func
        return decorator

# --- Setup for the Similar API: torch.library.impl_abstract (register_fake) ---
# We define a custom operator to simulate a scenario where an operator
# needs a specific abstract implementation (fake impl) to work with torch.compile
# on the MPS backend.

if HAS_NEW_API:
    lib = Library("test_mps_backend", "DEFINITION")
    lib.define("custom_mps_op(Tensor x) -> Tensor")

    # Register the abstract implementation (FakeTensor implementation).
    # This corresponds to torch.library.impl_abstract / torch.library.register_fake.
    # This is crucial for torch.compile to infer shapes and devices without running data.
    @register_fake("test_mps_backend::custom_mps_op")
    def custom_mps_op_abstract(x):
        # The abstract impl must return a tensor with the correct properties (shape, dtype, device).
        # If this returns a CPU tensor while x is MPS, torch.compile will fail on MPS.
        return torch.empty_like(x)

    # Register the actual implementation for MPS
    @impl("test_mps_backend::custom_mps_op", "MPS")
    def custom_mps_op_mps(x):
        # A simple operation for demonstration: x * 2
        return x * 2.0

    # Register the actual implementation for CPU (for fallback/comparison)
    @impl("test_mps_backend::custom_mps_op", "CPU")
    def custom_mps_op_cpu(x):
        return x * 2.0

# --- Test Case Adapted from Original Bug Report ---

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        # Check if the custom op was successfully registered into torch.ops
        # In PyTorch 1.x, this will likely be False, triggering the fallback.
        self.has_custom_op = hasattr(torch.ops, 'test_mps_backend') and \
                             hasattr(torch.ops.test_mps_backend, 'custom_mps_op')

    def forward(self, x):
        # Use the custom operator that relies on the registered abstract impl
        if self.has_custom_op:
            x = torch.ops.test_mps_backend.custom_mps_op(x)
        else:
            # Fallback for environments where custom op registration is not supported (e.g., PyTorch 1.x)
            x = x * 2.0
        return self.fc(x)

BATCH_SIZE = 4
NUM_CLASSES = 5
LEARNING_RATE = 0.01
device = 'mps'  # Targeting the device mentioned in the bug report

model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

model = model.to(device)
model.train()

# Handle torch.compile availability (PyTorch 2.0+)
if not hasattr(torch, 'compile'):
    print("torch.compile is not available. Running in eager mode.")
    def compile_wrapper(func):
        return func
    torch.compile = compile_wrapper

# We keep torch.compile to verify that the registered abstract implementation
# allows the backward pass to succeed on the MPS backend.
@torch.compile
def train_step(images, labels):
    images = images.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # The original bug reported failure here during backward()
    loss.backward()
    
    optimizer.step()
    return loss

images = torch.randn(BATCH_SIZE, 10)
labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

# Run the test
if torch.backends.mps.is_available():
    try:
        loss = train_step(images, labels)
        print(f"Test Passed: Forward and Backward completed on MPS. Loss: {loss.item()}")
    except RuntimeError as e:
        print(f"Test Failed: {e}")
else:
    print("MPS backend not available, skipping test.")