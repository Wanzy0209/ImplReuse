import torch

class Config:
    def __repr__(self):
        return "Config()"

# Mimicking the pattern from the similar API (tf.keras.optimizers.serialize / xla_call_module_test.serialize)
# which involves a serialization function returning a tuple of (string/bytes, int)
def serialize(obj) -> tuple[str, int]:
    # In the original bug, repr() was called directly. 
    # Here we wrap it in a 'serialize' function to leverage the similar API's pattern.
    return repr(obj), 1

def forward(x, config):
    # Calling the user-defined serialize function on a non-constant user object
    # This preserves the original bug reproduction logic (tracing repr on custom object)
    # while reusing the structure of the similar API.
    serialized_str, version = serialize(config)
    return x * len(serialized_str) * version

config = Config()
x = torch.randn(2, 2)

# Test the original API (torch.compile) with the adapted logic
compiled = torch.compile(forward, fullgraph=True)
result = compiled(x, config)

# Assertion to verify execution
assert result.shape == (2, 2)