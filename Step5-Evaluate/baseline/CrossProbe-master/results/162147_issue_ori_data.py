```python
import tensorflow as tf
import math

class FastLearnedCellX3(tf.keras.layers.Layer):
    def __init__(self, D_in, H, D_out,
                 L_w1=12, L_w2=12, L_b2=12,
                 k1=3, k2=3, k3=3, tau1=1.0, tau2=1.0, tau3=1.0,
                 d_addr=64,           # address bottleneck
                 learn_addr=False,
                 learn_tape_w1=True, learn_tape_w2=True, learn_tape_b2=True):
        super().__init__()
        self.D_in, self.H, self.D_out = D_in, H, D_out

        # keep sizes as plain Python ints (compile-friendly)
        self.L_w1, self.L_w2, self.L_b2 = int(L_w1), int(L_w2), int(L_b2)
        self.k1, self.k2, self.k3 = int(k1), int(k2), int(k3)
        self.t1, self.t2, self.t3 = float(tau1), float(tau2), float(tau3)

        # address bottleneck
        # Conversion: nn.Linear -> Dense
        # Conversion: torch.no_grad init -> trainable=False + custom initializer
        if learn_addr:
            self.P = tf.keras.layers.Dense(d_addr, use_bias=False, name='P')
        else:
            # Custom initializer for the specific normal distribution
            def init_p(shape, dtype=None):
                return tf.random.normal(shape, mean=0.0, stddev=1.0/math.sqrt(D_in), dtype=dtype)
            self.P = tf.keras.layers.Dense(d_addr, use_bias=False, kernel_initializer=init_p, trainable=False, name='P')

        def init_U(L, d, learn, name):
            # Conversion: torch.randn -> tf.random.normal
            U = tf.random.normal((L, d))
            # Conversion: mean(dim=1, keepdim=True) -> reduce_mean(..., axis=1, keepdims=True)
            U = U - tf.reduce_mean(U, axis=1, keepdims=True)
            # Conversion: norm(dim=1, keepdim=True) -> norm(..., axis=1, keepdims=True)
            U = U / (tf.norm(U, axis=1, keepdims=True) + 1e-8)
            # Conversion: nn.Parameter -> add_weight
            return self.add_weight(name=name, shape=(L, d), initializer=tf.constant_initializer(U), trainable=learn, dtype=tf.float32)

        # three unembeddings in addr-space
        self.U1 = init_U(self.L_w1, d_addr, learn_addr, 'U1')
        self.U2 = init_U(self.L_w2, d_addr, learn_addr, 'U2')
        self.U3 = init_U(self.L_b2, d_addr, learn_addr, 'U3')

        # value tapes
        # Conversion: F.normalize -> tf.math.l2_normalize
        def init_W(L, *shape_dims, learn, name):
            full_shape = (L,) + shape_dims
            # Conversion: torch.randn -> tf.random.normal
            W = tf.random.normal(full_shape)
            # Normalize over specified dimensions
            axes = tuple(range(1, len(full_shape)))
            W = tf.math.l2_normalize(W, axis=axes)
            return self.add_weight(name=name, shape=full_shape, initializer=tf.constant_initializer(W), trainable=learn, dtype=tf.float32)

        self.W1 = init_W(self.L_w1, H, D_in, learn=learn_tape_w1, name='W1')
        self.W2 = init_W(self.L_w2, D_out, H, learn=learn_tape_w2, name='W2')
        self.b2 = init_W(self.L_b2, D_out, learn=learn_tape_b2, name='b2')

        # Conversion: nn.GELU -> tf.nn.gelu
        self.act = tf.nn.gelu

    @staticmethod
    def _tk(z, k, tau):
        # Conversion: torch.topk -> tf.math.top_k
        topv, topi = tf.math.top_k(z, k=k, sorted=False)
        # Conversion: torch.softmax -> tf.nn.softmax
        w = tf.nn.softmax(topv / (tau + 1e-8), axis=1)
        return topi, w

    def _address(self, x_addr):
        # Conversion: torch.cat -> tf.concat
        U_pack = tf.concat([self.U1, self.U2, self.U3], axis=0)          # [Ltot, d]
        # Conversion: @ -> tf.matmul, .t() -> transpose_b=True
        Z = tf.linalg.matmul(x_addr, U_pack, transpose_b=True)           # [N, Ltot]
        s1, s2, s3 = self.L_w1, self.L_w2, self.L_b2
        # Conversion: torch.split -> tf.split
        z1, z2, z3 = tf.split(Z, [s1, s2, s3], axis=1)
        i1, w1 = FastLearnedCellX3._tk(z1, self.k1, self.t1)
        i2, w2 = FastLearnedCellX3._tk(z2, self.k2, self.t2)
        i3, w3 = FastLearnedCellX3._tk(z3, self.k3, self.t3)
        return (i1, w1), (i2, w2), (i3, w3)

    @staticmethod
    def _apply_mixture(x_flat, topi, weights, W):
        """
        Mix-then-apply (avoids replicating x):
        x_flat : [N, in]
        topi   : [N, k]
        weights: [N, k]
        W      : [L, out, in]
        return : [N, out]
        """
        N = tf.shape(topi)[0]
        k = topi.shape[1]
        out_dim = W.shape[1]
        in_dim = W.shape[2]
        
        # Conversion: index_select -> tf.gather
        # gather -> [N, k, out, in]
        W_sel = tf.gather(W, tf.reshape(topi, [-1]), axis=0)
        W_sel = tf.reshape(W_sel, [N, k, out_dim, in_dim])
        
        # mix neighbors -> [N, out, in]
        # Conversion: unsqueeze -> expand_dims
        W_eff = tf.reduce_sum(W_sel * tf.expand_dims(tf.expand_dims(weights, -1), -1), axis=1)
        
        # apply -> [N, out]
        # Conversion: bmm -> tf.matmul
        # x_flat: [N, in] -> [N, in, 1]
        # W_eff: [N, out, in]
        # result: [N, out, 1]
        res = tf.linalg.matmul(W_eff, tf.expand_dims(x_flat, -1))
        return tf.squeeze(res, axis=-1)

    def call(self, x):
        # default values for TorchScript
        # Conversion: x.ndim -> len(x.shape) (static shape check)
        input_shape = tf.shape(x)
        static_rank = len(x.shape)
    
        if static_rank == 3:
            B = input_shape[0]
            T = input_shape[1]
            D = input_shape[2]
            # Conversion: reshape -> tf.reshape
            x_flat = tf.reshape(x, [-1, D])
        else:
            B = input_shape[0]
            T = 1  # treat as sequence length 1
            x_flat = x
        
        x_addr = self.P(x_flat)                                    # [N, d_addr]
        (i1, w1), (i2, w2), (i3, w3) = self._address(x_addr)

        # FC1
        h = self._apply_mixture(x_flat, i1, w1, self.W1)           # [N, H]
        h = self.act(h)

        # FC2 + bias
        y = self._apply_mixture(h, i2, w2, self.W2)                # [N, D_out]
        
        # Conversion: index_select -> tf.gather
        b = tf.gather(self.b2, tf.reshape(i3, [-1]), axis=0)
        b = tf.reshape(b, [tf.shape(x_flat)[0], self.k3, self.D_out])
        
        y = y + tf.reduce_sum(tf.expand_dims(w3, -1) * b, axis=1)
    
        # Now both branches have valid B and T
        if static_rank == 3:
            return tf.reshape(y, [B, T, self.D_out])
        else:
            return y
```