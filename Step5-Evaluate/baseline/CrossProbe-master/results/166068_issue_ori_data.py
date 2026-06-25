```python
import tensorflow as tf
from packaging import version

# PyTorch uses torch.version.hip to check the HIP runtime version.
# TensorFlow exposes build configuration details via tf.sysconfig.
# We retrieve 'rocm_version' as the closest equivalent, defaulting to 'None' if not found.
print(version.parse(tf.sysconfig.get_build_info().get('rocm_version', 'None')))
```