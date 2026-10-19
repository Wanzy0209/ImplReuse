```python
import tensorflow as tf
from packaging import version

# Conversion: torch.version.hip provides the ROCm version string.
# TensorFlow provides this via tf.sysconfig.get_build_info() under the 'rocm_version' key.
print(version.parse(tf.sysconfig.get_build_info().get('rocm_version')))
```