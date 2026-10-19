```python
import tensorflow as tf
from tensorflow.python.training.saving.saveable_object_util import trackable_has_serialize_to_tensor

class JITCrash(tf.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1

model = JITCrash()
# Conversion: torch.jit.script(model) is mapped to trackable_has_serialize_to_tensor(model)
# based on the provided documentation context.
scripted = trackable_has_serialize_to_tensor(model)
```