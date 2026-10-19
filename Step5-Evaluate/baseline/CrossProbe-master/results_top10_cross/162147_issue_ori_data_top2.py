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
        self.D_in, self.H, self.D_out = D_in, H, D_out

        # keep sizes as plain Python ints (compile-friendly)
        self.L_w1, self.L_w2, self.L_b2 = int(L_w1), int(L_w2), int(L_b2)
        self.k1, self.k2, self.k3 = int(k1), int(k2), int(k3)
        self.t1, self.t2, self.t3 = float(tau1), float(tau2), float(tau3)

        # address bottleneck
        # Conversion: nn.Linear -> Dense layer logic (weights stored as [in, out] in TF vs [out, in] in PT)
        # We use add_weight to mimic the parameter structure closely.
        self.P = self.add_weight(name="P_weight",
                                 shape=[D_in, d_addr],
                                 initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(D_in)),
                                 trainable=learn_addr)

        def init_U(L, d, learn):
            # Conversion: torch.randn -> tf.random.normal
            U = tf.random.normal([L, d])
            # Conversion: U.mean(dim=1, keepdim=True) -> tf.reduce_mean(..., axis=1, keepdims=True)
            U = U - tf.reduce_mean(U, axis=1, keepdims=True)
            # Conversion: U.norm(...) -> tf.norm(...)
            U = U / (tf.norm(U, axis=1, keepdims=True) + 1e-8)
            return self.add_weight(name=f"U_{L}", shape=[L, d], 
                                   initializer=tf.constant_initializer(U), 
                                   trainable=learn)

        # three unembeddings in addr-space
        self.U1 = init_U(self.L_w1, d_addr, learn_addr)
        self.U2 = init_U(self.L_w2, d_addr, learn_addr)
        self.U3 = init_U(self.L_b2, d_addr, learn_addr)

        # value tapes
        # Conversion: F.normalize -> tf.math.l2_normalize
        # Note: F.normalize in PT normalizes over specified dims. 
        # We pre-calculate the normalized values for the initializer.
        
        # W1: [L_w1, H, D_in]
        W1_val = tf.random.normal([self.L_w1, H, D_in])
        W1_val = tf.math.l2_normalize(W1_val, axis=(1,2))
        self.W1 = self.add_weight(name="W1", shape=[self.L_w1, H, D_in],
                                  initializer=tf.constant_initializer(W1_val),
                                  trainable=learn_tape_w1)

        # W2: [L_w2, D_out, H]
        W2_val = tf.random.normal([self.L_w2, D_out, H])
        W2_val = tf.math.l2_normalize(W2_val, axis=(1,2))
        self.W2 = self.add_weight(name="W2", shape=[self.L_w2, D_out, H],
                                  initializer=tf.constant_initializer(W2_val),
                                  trainable=learn_tape_w2)

        # b2: [L_b2, D_out]
        b2_val = tf.random.normal([self.L_b2, D_out])
        b2_val = tf.math.l2_normalize(b2_val, axis=1)
        self.b2 = self.add_weight(name="b2", shape=[self.L_b2, D_out],
                                  initializer=tf.constant_initializer(b2_val),
                                  trainable=learn_tape_b2)

        self.act = tf.nn.gelu

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
        
        # Conversion: x @ U.t() -> tf.matmul(x, U, transpose_b=True)
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
        # Conversion: W.index_select(0, topi.reshape(-1)) -> tf.gather(W, tf.reshape(topi, [-1]), axis=0)
        W_sel = tf.gather(W, tf.reshape(topi, [-1]), axis=0)
        W_sel = tf.reshape(W_sel, [N, k, out_dim, in_dim])
        
        # mix neighbors -> [N, out, in]
        # Conversion: weights.unsqueeze(-1).unsqueeze(-1) -> tf.reshape(weights, [N, k, 1, 1])
        W_eff = tf.reduce_sum(W_sel * tf.reshape(weights, [N, k, 1, 1]), axis=1)
        
        # apply -> [N, out]
        # Conversion: torch.bmm -> tf.matmul
        # x_flat.unsqueeze(-1) -> tf.expand_dims(x_flat, -1)
        res = tf.matmul(W_eff, tf.expand_dims(x_flat, -1))
        return tf.squeeze(res, -1)

    def call(self, x):
        # Conversion: x.ndim -> x.shape.rank
        if x.shape.rank == 3:
            B = tf.shape(x)[0]
            T = tf.shape(x)[1]
            D = tf.shape(x)[2]
            x_flat = tf.reshape(x, [B * T, D])
        else:
            B = tf.shape(x)[0]
            T = 1  # treat as sequence length 1
            x_flat = x
        
        # Conversion: self.P(x_flat) -> tf.matmul(x_flat, self.P)
        # Note: self.P is [D_in, d_addr], x_flat is [N, D_in]
        x_addr = tf.matmul(x_flat, self.P)                             # [N, d_addr]
        (i1, w1), (i2, w2), (i3, w3) = self._address(x_addr)

        # FC1
        h = self._apply_mixture(x_flat, i1, w1, self.W1)               # [N, H]
        h = self.act(h)

        # FC2 + bias
        y = self._apply_mixture(h, i2, w2, self.W2)                    # [N, D_out]
        
        # Bias handling
        # Conversion: self.b2.index_select(0, i3.reshape(-1)) -> tf.gather(self.b2, tf.reshape(i3, [-1]), axis=0)
        b = tf.gather(self.b2, tf.reshape(i3, [-1]), axis=0)
        b = tf.reshape(b, [tf.shape(x_flat)[0], self.k3, self.D_out])
        
        # Conversion: (w3.unsqueeze(-1) * b).sum(dim=1) -> tf.reduce_sum(tf.expand_dims(w3, -1) * b, axis=1)
        y = y + tf.reduce_sum(tf.expand_dims(w3, -1) * b, axis=1)
    
        # Now both branches have valid B and T
        # Conversion: y.view(B, T, self.D_out) -> tf.reshape(y, [B, T, self.D_out])
        if x.shape.rank == 3:
            return tf.reshape(y, [B, T, self.D_out])
        else:
            return y
```