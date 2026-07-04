```python
import tensorflow_probability as tfp

# Conversion Note: TensorFlow Probability does not support a global setting to disable
# argument validation for all distributions. To achieve the same effect, pass
# `validate_args=False` explicitly to the constructor of each distribution.
# torch.distributions.Distribution.set_default_validate_args(False)
```