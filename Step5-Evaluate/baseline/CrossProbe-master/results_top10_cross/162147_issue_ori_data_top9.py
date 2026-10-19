```python
import tensorflow as tf
import math
from typing import Tuple

class FastLearnedCellX3(tf.keras.layers.Layer):
    def __init__(self, D_in, H, D_out,
                 L_w1=12, L_w2=12, L_b2=12,
                 k1=3, k2=3, k3=3, tau1=1.0, tau2=1.0, tau3=1.0,
                 d_addr=64,           # address bottleneck
                 learn_addr=False,
                 learn_tape_w1=True, learn_tape_w2=True, learn_tape_b2=True):
        super().__init__()
        self.D_in, self.H, self.D_out = int(D_in), int(H), int(D_out)

        # keep sizes as plain Python ints (compile-friendly)
        self.L_w1, self.L_w2, self.L_b2 = int(L_w1), int(L_w2), int(L_b2)
        self.k1, self.k2, self.k3 = int(k1), int(k2), int(k3)
        self.t1, self.t2, self.t3 = float(tau1), float(tau2), float(tau3)

        # address bottleneck
        # Conversion: nn.Linear -> Dense
        self.P = tf.keras.layers.Dense(d_addr, use_bias=False, name='P')
        if not learn_addr:
            self.P.trainable = False
            # Conversion: torch.no_grad context not needed in TF init, 
            # but we set the initializer explicitly to match the logic.
            self.P.kernel_initializer = tf.keras.initializers.RandomNormal(stddev=1.0/math.sqrt(D_in))

        # Helper to create normalized random weights (U)
        def init_U(shape, dtype):
            U = tf.random.normal(shape, dtype=dtype)
            U = U - tf.reduce_mean(U, axis=1, keepdims=True)
            norm = tf.norm(U, axis=1, keepdims=True)
            return U / (norm + 1e-8)

        # three unembeddings in addr-space
        # Conversion: nn.Parameter -> add_weight
        self.U1 = self.add_weight(name='U1', shape=(self.L_w1, d_addr), initializer=init_U, trainable=learn_addr)
        self.U2 = self.add_weight(name='U2', shape=(self.L_w2, d_addr), initializer=init_U, trainable=learn_addr)
        self.U3 = self.add_weight(name='U3', shape=(self.L_b2, d_addr), initializer=init_U, trainable=learn_addr)

        # Helper to create normalized random weights (W)
        def init_W(shape, dtype):
            W = tf.random.normal(shape, dtype=dtype)
            # Normalize over dim (1,2) -> last 2 dims
            L = shape[0]
            W_flat = tf.reshape(W, [L, -1])
            W_norm = tf.linalg.l2_normalize(W_flat, axis=1)
            return tf.reshape(W_norm, shape)

        # value tapes
        self.W1 = self.add_weight(name='W1', shape=(self.L_w1, self.H, self.D_in), initializer=init_W, trainable=learn_tape_w1)
        self.W2 = self.add_weight(name='W2', shape=(self.L_w2, self.D_out, self.H), initializer=init_W, trainable=learn_tape_w2)
        
        # Helper for b2 (normalize over dim 1)
        def init_b(shape, dtype):
            b = tf.random.normal(shape, dtype=dtype)
            return tf.linalg.l2_normalize(b, axis=1)

        self.b2 = self.add_weight(name='b2', shape=(self.L_b2, self.D_out), initializer=init_b, trainable=learn_tape_b2)

        self.act = tf.keras.layers.Activation('gelu')

    @staticmethod
    def _tk(z: tf.Tensor, k: int, tau: float) -> Tuple[tf.Tensor, tf.Tensor]:
        # Conversion: torch.topk -> tf.math.top_k
        topv, topi = tf.math.top_k(z, k=k, sorted=False)
        # Conversion: torch.softmax -> tf.nn.softmax
        w = tf.nn.softmax(topv / (tau + 1e-8), axis=1)
        return topi, w

    def _address(self, x_addr: tf.Tensor):
        # fused logits for all three heads
        # Conversion: torch.cat -> tf.concat
        U_pack = tf.concat([self.U1, self.U2, self.U3], axis=0)          # [Ltot, d]
        # Conversion: @ -> tf.matmul, .t() -> transpose_b=True
        Z = tf.matmul(x_addr, U_pack, transpose_b=True)                 # [N, Ltot]
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
        k = tf.shape(topi)[1]
        out_dim = tf.shape(W)[1]
        in_dim = tf.shape(W)[2]
        
        # gather -> [N, k, out, in]
        # Conversion: index_select -> tf.gather
        topi_flat = tf.reshape(topi, [-1])
        W_sel = tf.gather(W, topi_flat, axis=0)
        W_sel = tf.reshape(W_sel, [N, k, out_dim, in_dim])
        
        # mix neighbors -> [N, out, in]
        # Conversion: unsqueeze -> expand_dims
        weights_exp = tf.expand_dims(tf.expand_dims(weights, -1), -1)
        W_eff = tf.reduce_sum(W_sel * weights_exp, axis=1)
        
        # apply -> [N, out]
        # Conversion: bmm -> tf.matmul (handles batch matmul)
        x_exp = tf.expand_dims(x_flat, -1) # [N, in, 1]
        res = tf.matmul(W_eff, x_exp)      # [N, out, 1]
        return tf.squeeze(res, -1)

    def _compute_core(self, x_flat):
        x_addr = self.P(x_flat)                                    # [N, d_addr]
        (i1, w1), (i2, w2), (i3, w3) = self._address(x_addr)

        # FC1
        h = self._apply_mixture(x_flat, i1, w1, self.W1)           # [N, H]
        h = self.act(h)

        # FC2 + bias
        y = self._apply_mixture(h, i2, w2, self.W2)                # [N, D_out]
        
        # Bias calculation
        # Conversion: index_select -> tf.gather
        b_sel = tf.gather(self.b2, tf.reshape(i3, [-1]), axis=0)
        b_sel = tf.reshape(b_sel, [tf.shape(x_flat)[0], self.k3, self.D_out])
        
        y = y + tf.reduce_sum(w3[..., tf.newaxis] * b_sel, axis=1)
        
        return y

    def call(self, x):
        # Conversion: Handling dynamic rank with tf.cond for graph compatibility
        rank = tf.rank(x)
        
        def run_3d():
            # B, T, D = x.shape
            s = tf.shape(x)
            B, T, D = s[0], s[1], s[2]
            x_flat = tf.reshape(x, [B * T, D])
            y = self._compute_core(x_flat)
            return tf.reshape(y, [B, T, self.D_out])

        def run_2d():
            # B, D = x.shape
            return self._compute_core(x)

        # If rank is 3, treat as batched sequence, else as single sample/batch
        return tf.cond(tf.equal(rank, 3), run_3d, run_2d)
```