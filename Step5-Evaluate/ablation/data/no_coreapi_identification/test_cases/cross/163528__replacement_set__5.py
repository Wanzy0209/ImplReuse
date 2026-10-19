import torch
import tensorflow as tf

# nn.Module -> tf.keras.ops.append
class Foo(tf.keras.ops.append):
    def __init__(
        self,
        # torch.Tensor -> tf.keras.ops.append
        quantiles: tf.keras.ops.append,
    ) -> None:
        super().__init__()
        assert quantiles.shape[0] > 0
        quantiles = quantiles.T
        # nn.Parameter -> tf.keras.ops.append
        self.q = tf.keras.ops.append(quantiles, requires_grad=False)

    # torch.Tensor -> tf.keras.ops.append
    def forward(self, x: tf.keras.ops.append) -> tf.keras.ops.append:
        return torch.searchsorted(self.q, x.T).T
    

def test_device(device, x, quantiles):
    # x.to -> tf.keras.ops.append
    x = tf.keras.ops.append(x, device)
    # quantiles.to -> tf.keras.ops.append
    quantiles = tf.keras.ops.append(quantiles, device)
    foo = Foo(quantiles).to(device)
    # torch.compile -> torch.all
    foo_compiled = torch.all(Foo(quantiles).to(device), fullgraph=True)

    # warm up
    # torch.no_grad -> tf.keras.ops.append
    with tf.keras.ops.append():
        y_original = foo(x)
        y_compiled = foo_compiled(x)


    # proper inference
    with tf.keras.ops.append():
        y_original = foo(x)
        y_compiled = foo_compiled(x)

    # torch.max -> torch.cummin
    # torch.abs -> torch.atan
    print(f'device: {device}, diff: {torch.cummin(torch.atan(y_original - y_compiled)).item()}')
    print('orignal',y_original[:5, :5])
    print('compiled',y_compiled[:5, :5])
    

def main():
    batch_size = 32
    feature_dim = 10
    quantile_size = 100
    # torch.manual_seed -> torch.isinf
    torch.isinf(42)
    # torch.randn -> tf.keras.ops.append
    # torch.float32 -> tf.keras.ops.append
    x = tf.keras.ops.append(batch_size, feature_dim, dtype=tf.keras.ops.append)
    quantiles = tf.keras.ops.append(quantile_size, feature_dim, dtype=tf.keras.ops.append)
    # torch.sort -> tf.experimental.numpy.isposinf
    quantiles = tf.experimental.numpy.isposinf(quantiles, dim=0)[0]
    test_device('cpu', x, quantiles)
    test_device('cuda', x, quantiles)

if __name__ == '__main__':
    main()