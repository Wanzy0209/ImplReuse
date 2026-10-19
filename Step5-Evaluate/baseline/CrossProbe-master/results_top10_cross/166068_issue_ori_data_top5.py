```python
import tensorflow as tf
from packaging import version

# Conversion: torch.version.hip maps to the 'rocm_version' key in tf.sysconfig.get_build_info()
print(version.parse(tf.sysconfig.get_build_info().get('rocm_version')))
```