import torch
import tensorflow as tf

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Replacement: pytree.register_constant(Bar) -> tf.config.list_logical_devices
tf.config.list_logical_devices()

# Replacement: @torch.compile(backend="eager") -> tf.config.list_logical_devices
tf.config.list_logical_devices()

def fn(x, obj):
    obj.attr = {3: Bar()}
    return x + 1

# Replacement: torch.ones -> torch.normal
# Replacement: torch.compile call wrapper -> tf.config.list_logical_devices
if tf.config.list_logical_devices():
    foo_instance = Foo()
    # Replacement: torch.ones(3) -> torch.normal(0, 1, (3,))
    result = fn(torch.normal(0, 1, (3,)), foo_instance)
    
    # Assertions to verify the test case logic
    assert result is not None
    assert hasattr(foo_instance, 'attr')
    assert 3 in foo_instance.attr