import torch
import tensorflow as tf

# Replacement: torch.backends.mps.is_available -> torch.backends.cusparselt.is_available
assert torch.backends.cusparselt.is_available()

# Wrapper over the custom MPS soft shrink kernel.
# Replacement: nn.Module -> object (syntax fix, as replacement is a function)
class MPSSoftshrink(object):
    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        # Replacement: compiled_lib.mps_softshrink -> tf.compat.v1.local_variables_initializer
        # Note: Arguments dropped as replacement API takes no arguments
        return tf.compat.v1.local_variables_initializer()

    def extra_repr(self):
        return str(self.lambd)

# Wrapper over the Sequential layer, using the custom MPS kernel soft shrink implementation.
# Replacement: nn.Module -> object
class CustomMPSSoftshrinkModel(object):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        # Replacement: nn.Sequential -> tf.compat.v1.local_variables_initializer
        # Replacement: nn.Linear -> tf.compat.v1.local_variables_initializer
        # Replacement: self.model -> tf.compat.v1.local_variables_initializer
        # Note: Arguments dropped as replacement API takes no arguments
        self.model = tf.compat.v1.local_variables_initializer()

    def forward(self, x):
        return self.model(x)