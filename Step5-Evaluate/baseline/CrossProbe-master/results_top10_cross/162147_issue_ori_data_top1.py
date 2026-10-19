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
        # Conversion: nn.Linear -> tf.keras.layers.Dense
        self.P = tf.keras.layers.Dense(d_addr, use_bias=False)
        if not learn_addr:
            self.P.trainable = False
            # Conversion: nn.init.normal_ -> setting kernel_initializer
            self.P.kernel_initializer = tf.random_normal_initializer(stddev=1.0/math.sqrt(D_in))

        def init_U(L, d, learn):
            # Conversion: torch.randn -> tf.random.normal
            U = tf.random.normal((L, d))
            U = U - tf.reduce_mean(U, axis=1, keepdims=True)
            # Conversion: U.norm -> tf.linalg.norm
            U = U / (tf.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
            # Conversion: nn.Parameter -> self.add_weight
            return self.add_weight(name=f'U_{L}', shape=U.shape, 
                                   initializer=tf.constant_initializer(U), 
                                   trainable=learn)

        # three unembeddings in addr-space
        self.U1 = init_U(self.L_w1, d_addr, learn_addr)
        self.U2 = init_U(self.L_w2, d_addr, learn_addr)
        self.U3 = init_U(self.L_b2, d_addr, learn_addr)

        # value tapes
        # Conversion: F.normalize -> tf.nn.l2_normalize
        W1_raw = tf.random.normal((self.L_w1, H, D_in))
        W1_norm = tf.nn.l2_normalize(W1_raw, axis=(1,2))
        self.W1 = self.add_weight(name='W1', shape=W1_norm.shape,
                                  initializer=tf.constant_initializer(W1_norm),
                                  trainable=learn_tape_w1)

        W2_raw = tf.random.normal((self.L_w2, D_out, H))
        W2_norm = tf.nn.l2_normalize(W2_raw, axis=(1,2))
        self.W2 = self.add_weight(name='W2', shape=W2_norm.shape,
                                  initializer=tf.constant_initializer(W2_norm),
                                  trainable=learn_tape_w2)

        b2_raw = tf.random.normal((self.L_b2, D_out))
        b2_norm = tf.nn.l2_normalize(b2_raw, axis=1)
        self.b2 = self.add_weight(name='b2', shape=b2_norm.shape,
                                  initializer=tf.constant_initializer(b2_norm),
                                  trainable=learn_tape_b2)

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
        # fused logits for all three heads
        # Conversion: torch.cat -> tf.concat
        U_pack = tf.concat([self.U1, self.U2, self.U3], axis=0)          # [Ltot, d]
        # Conversion: @ -> tf.linalg.matmul with transpose_b
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
        
        # gather -> [N, k, out, in]
        # Conversion: index_select -> tf.gather
        W_sel = tf.gather(W, tf.reshape(topi, [-1]), axis=0)
        W_sel = tf.reshape(W_sel, [N, k, out_dim, in_dim])
        
        # mix neighbors -> [N, out, in]
        # Conversion: unsqueeze -> tf.expand_dims
        W_eff = tf.reduce_sum(W_sel * tf.expand_dims(tf.expand_dims(weights, -1), -1), axis=1)
        
        # apply -> [N, out]
        # Conversion: torch.bmm -> tf.matmul
        return tf.squeeze(tf.matmul(W_eff, tf.expand_dims(x_flat, -1)), -1)

    def call(self, x):
        # default values for TorchScript
        B = 1   # dummy placeholder
        T = 1
    
        # Conversion: x.ndim -> x.shape.ndims
        if x.shape.ndims == 3:
            B = tf.shape(x)[0]
            T = tf.shape(x)[1]
            D = x.shape[2]
            x_flat = tf.reshape(x, [B * T, D])
        else:
            B = tf.shape(x)[0]
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
        # Conversion: view -> tf.reshape
        if x.shape.ndims == 3:
            return tf.reshape(y, [B, T, self.D_out])
        else:
            return y
```