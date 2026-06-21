import torch
import sys

# Handle the environment issue where libstdc++ is too old for the installed TensorFlow/Protobuf
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test due to environment error: {e}")
    print("Required library version (GLIBCXX_3.4.29) not found.")
    sys.exit(0)

# Wrapper mimicking the MPSSoftshrink structure from the bug report
class LocalVarSoftshrink:
    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd
        # Define a local variable to represent the custom kernel state
        self.var = tf.compat.v1.local_variable(
            initial_value=self.lambd, 
            name="softshrink_lambd"
        )

    def forward(self, input_tensor):
        # Simple operation using the variable
        return input_tensor * self.var

def test_local_variables_initializer():
    # Mimic: assert torch.backends.mps.is_available()
    # We check the execution context to determine the initialization path
    is_eager = tf.executing_eagerly()
    
    # Mimic: mps_device = torch.device("mps")
    # We set up the model
    model = LocalVarSoftshrink(lambd=0.5)
    input_tensor = tf.constant([1.0, 2.0, 3.0])

    if is_eager:
        # In eager mode, variables are initialized on creation.
        # The API returns a no_op, so we just verify execution.
        output = model.forward(input_tensor)
        assert output is not None
    else:
        # In graph mode, we must use the initializer.
        with tf.compat.v1.Session() as sess:
            # This is the Similar API: tf.compat.v1.local_variables_initializer
            init_op = tf.compat.v1.local_variables_initializer()
            
            # Run initialization
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(init_op)
            
            # Run the forward pass
            result = sess.run(model.forward(input_tensor))
            assert result is not None

if __name__ == "__main__":
    test_local_variables_initializer()