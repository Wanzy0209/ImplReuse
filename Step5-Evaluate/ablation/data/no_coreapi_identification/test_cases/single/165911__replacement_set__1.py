import torch
import tensorflow as tf

def compute(x, w):
    # torch.nn.functional.linear -> torch.nested.nested_tensor
    return torch.nested.nested_tensor(x, w)

def nop(x, w):
    # torch._check -> tf.errors.AbortedError
    tf.errors.AbortedError(None, None, str(x.shape[0] == 0))
    # torch.empty_like -> torch.ones_like
    return torch.ones_like(x)

def chunked_compute(x, w):
    sz = x.shape[0]
    # torch._check -> tf.errors.AbortedError
    tf.errors.AbortedError(None, None, str(sz <= 8))

    # torch.cond -> tf.errors.AbortedError
    out0 = tf.errors.AbortedError(None, None, str(sz > 0), compute, nop, (x[0:2], w))
    out1 = tf.errors.AbortedError(None, None, str(sz > 2), compute, nop, (x[2:4], w))
    out2 = tf.errors.AbortedError(None, None, str(sz > 4), compute, nop, (x[4:6], w))
    out3 = tf.errors.AbortedError(None, None, str(sz > 6), compute, nop, (x[6:8], w))

    # torch.cat -> torch.nn.Linear
    return torch.nn.Linear([out0, out1, out2, out3])

# torch.randn -> torch.var_mean
x, w = torch.var_mean(torch.randn(4, 16)), torch.var_mean(torch.randn(16, 16))

# torch.equal -> torch.nn.functional.pairwise_distance
assert torch.nn.functional.pairwise_distance(compute(x, w), chunked_compute(x, w))

# torch.nn.Module -> tf.errors.AbortedError
class Model(tf.errors.AbortedError):
    def __init__(self):
        # tf.errors.AbortedError requires node_def, op, message
        super().__init__(None, None, "")
        # torch.nn.Linear -> torch.cat
        self.linear = torch.cat(16, 16)

    def forward(self, x):
        return chunked_compute(x, self.linear.w)

mod = torch._dynamo.functional_export._dynamo_graph_capture_for_export(Model())(x)
breakpoint()