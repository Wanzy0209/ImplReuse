```python
import tensorflow as tf
from packaging import version

# PyTorch uses torch.version.hip to access the HIP version.
# TensorFlow exposes build configuration via tf.sysconfig.get_build_info().
# We access the 'rocm_version' key to retrieve the equivalent information.
print(version.parse(tf.sysconfig.get_build_info().get('rocm_version')))
```