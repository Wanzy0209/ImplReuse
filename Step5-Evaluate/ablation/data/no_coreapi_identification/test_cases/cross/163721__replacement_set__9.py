import torch
import tensorflow as tf

# Replacement: torch.backends.mps.is_available -> torch.backends.opt_einsum.get_opt_einsum
# Original: assert torch.backends.mps.is_available()
# New: Check opt_einsum strategy availability
strategy = torch.backends.opt_einsum.get_opt_einsum()
assert strategy is not None

# Replacement: nn.Module -> tf.IndexedSlices
class MPSSoftshrink(tf.IndexedSlices):
    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        # tf.IndexedSlices requires values and indices tensors.
        # We provide dummy tensors to satisfy the constructor for runnability.
        super().__init__(values=tf.constant([0.0]), indices=tf.constant([0]))
        self.lambd = lambd

    def forward(self, input):
        # Replacement: compiled_lib.mps_softshrink -> tf.IndexedSlices
        # Original: return compiled_lib.mps_softshrink(input, self.lambd)
        # New: Return an IndexedSlices object.
        # Translating semantics: input becomes values, lambd becomes indices (as a tensor).
        return tf.IndexedSlices(values=input, indices=tf.constant([self.lambd]))

    def extra_repr(self):
        return str(self.lambd)

# Replacement: nn.Module -> tf.IndexedSlices
class CustomMPSSoftshrinkModel(tf.IndexedSlices):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        # tf.IndexedSlices requires values and indices tensors.
        super().__init__(values=tf.constant([0.0]), indices=tf.constant([0]))

        # Replacement: nn.Sequential -> tf.IndexedSlices
        # Replacement: self.model -> tf.IndexedSlices
        # Original: self.model = nn.Sequential(...)
        # New: self.model = tf.IndexedSlices(...)
        # Note: The original nn.Sequential contained a list of layers (nn.Linear, MPSSoftshrink).
        # Since tf.IndexedSlices is a data structure and not a container for layers,
        # we instantiate it with dummy tensors to maintain runnability.
        self.model = tf.IndexedSlices(values=tf.constant([0.0]), indices=tf.constant([0]))

    def forward(self, x):
        return self.model

# Test execution to ensure runnability
if __name__ == "__main__":
    # Instantiate the custom model
    model = CustomMPSSoftshrinkModel()
    
    # Instantiate the custom shrink layer
    shrink_layer = MPSSoftshrink(lambd=0.5)
    
    # Create a dummy input tensor
    dummy_input = tf.constant([1.0, 2.0, 3.0])
    
    # Run forward pass on the shrink layer
    output = shrink_layer.forward(dummy_input)
    
    # Verify output type
    assert isinstance(output, tf.IndexedSlices)
    
    print("Test case executed successfully.")