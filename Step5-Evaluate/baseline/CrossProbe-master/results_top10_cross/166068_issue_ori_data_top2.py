```python
import tensorflow as tf
from packaging import version

# Conversion: PyTorch uses torch.version.hip for the ROCm version.
# TensorFlow uses tf.sysconfig.get_build_info() to access build details.
# We retrieve 'rocm_version' from the build info dictionary.
print(version.parse(tf.sysconfig.get_build_info().get('rocm_version')))
```