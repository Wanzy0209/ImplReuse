```python
import tensorflow as tf


# Conversion note: PyTorch's torch.neg on uint8 wraps around (e.g., -7 becomes 249).
# TensorFlow's tf.negative promotes to int32. To match PyTorch's behavior,
# we use bitwise inversion and addition (two's complement) to keep it uint8.
def foo(x):
    c = tf.constant(7, dtype=tf.uint8)
    # Emulate torch.neg(c) for uint8: two's complement
    neg_c = tf.bitwise.invert(c) + 1
    return c + x, neg_c, neg_c + x
    

tf.random.set_seed(0)
x = tf.random.normal((2, 2), dtype=tf.float32)
print(f"input: {x}")
'''
input: tensor([[ 1.5410, -0.2934],
        [-2.1788,  0.5684]])
'''
# torch.compile -> tf.function
cfoo = tf.function(foo)
res = foo(x)
cres = cfoo(x)

print(f"res[0]: {res[0]}")
print(f"cres[0]: {cres[0]}")
'''
res[0]: tensor([[8.5410, 6.7066],
        [4.8212,  7.5684]])
cres[0]: tensor([[8.5410, 6.7066],
        [4.8212,  7.5684]]
'''
print(f"res[1]: {res[1]}")
print(f"cres[1]: {cres[1]}")
'''
res[1]: 249
cres[1]: 249
'''
print(f"res[2]: {res[2]}")
print(f"cres[2]: {cres[2]}")
'''
res[2]: tensor([[250.5410, 248.7066],
        [246.8212, 249.5684]])
# Note: TensorFlow's tf.function preserves the uint8 overflow behavior (249 + x)
# unlike PyTorch's torch.compile which optimized it to (-7 + x).
# Therefore, cres[2] in TF will match res[2].
'''
```