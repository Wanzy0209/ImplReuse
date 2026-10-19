```python
import tensorflow as tf
from packaging import version

# Conversion: torch.version.hip -> tf.sysconfig.get_build_info().get('rocm_version')
# PyTorch exposes the HIP version directly, while TensorFlow stores it in build info.
print(version.parse(tf.sysconfig.get_build_info().get('rocm_version')))
```