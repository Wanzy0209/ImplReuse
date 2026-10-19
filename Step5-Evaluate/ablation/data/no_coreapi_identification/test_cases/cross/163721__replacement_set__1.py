import torch
import tensorflow as tf

# Replacement: torch.backends.mps.is_available -> torch.distributed.is_nccl_available
assert torch.distributed.is_nccl_available()

# Replacement: nn.Module -> object (to maintain validity)
class MPSSoftshrink(object):
    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        self.lambd = lambd

    def forward(self, input):
        # Replacement: compiled_lib.mps_softshrink -> tf.compat.v1.global_variables_initializer
        return tf.compat.v1.global_variables_initializer(input, self.lambd)

    def extra_repr(self):
        return str(self.lambd)

class CustomMPSSoftshrinkModel(object):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        # Replacement: nn.Sequential -> tf.compat.v1.global_variables_initializer
        # Replacement: nn.Linear -> tf.compat.v1.global_variables_initializer
        # Replacement: self.model -> tf.compat.v1.global_variables_initializer
        tf.compat.v1.global_variables_initializer = tf.compat.v1.global_variables_initializer(
            tf.compat.v1.global_variables_initializer(input_size, lin1_size),
            MPSSoftshrink(),
            tf.compat.v1.global_variables_initializer(lin1_size, lin2_size),
            MPSSoftshrink(),
            tf.compat.v1.global_variables_initializer(lin2_size, lin3_size),
            MPSSoftshrink(),
            tf.compat.v1.global_variables_initializer(lin3_size, output_size),
        )

    def forward(self, x):
        # Replacement: self.model -> tf.compat.v1.global_variables_initializer
        return tf.compat.v1.global_variables_initializer(x)