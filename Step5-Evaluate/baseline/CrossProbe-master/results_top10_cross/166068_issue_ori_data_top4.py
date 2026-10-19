```python
import tensorflow as tf
from packaging import version

# Conversion: PyTorch exposes the ROCm version via torch.version.hip.
# TensorFlow requires accessing the build info dictionary to find the ROCm version.
print(version.parse(tf.sysconfig.get_build_info().get('rocm_version')))
```