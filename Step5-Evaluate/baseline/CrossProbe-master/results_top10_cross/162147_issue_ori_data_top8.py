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
        # PyTorch: nn.Linear(D_in, d_addr, bias=False)
        # Note: PyTorch default init is uniform, but custom init is applied if not learn_addr
        if not learn_addr:
            kernel_init = tf.random_normal_initializer(stddev=1.0/math.sqrt(D_in))
        else:
            kernel_init = 'glorot_uniform' # Default for Dense
            
        self.P = tf.keras.layers.Dense(
            d_addr, 
            use_bias=False, 
            kernel_initializer=kernel_init,
            name='P'
        )
        self.P.trainable = learn_addr

        def init_U(L, d, learn, name):
            # PyTorch: torch.randn(L, d)
            U = tf.random.normal((L, d))
            # PyTorch: U - U.mean(dim=1, keepdim=True)
            U = U - tf.reduce_mean(U, axis=1, keepdims=True)
            # PyTorch: U / (U.norm(dim=1, keepdim=True) + 1e-8)
            U = U / (tf.norm(U, axis=1, keepdims=True) + 1e-8)
            # PyTorch: nn.Parameter(U, requires_grad=learn)
            return self.add_weight(
                name=name, 
                shape=U.shape, 
                dtype=tf.float32,
                initializer=tf.constant_initializer(U),
                trainable=learn
            )

        # three unembeddings in addr-space
        self.U1 = init_U(self.L_w1, d_addr, learn_addr, 'U1')
        self.U2 = init_U(self.L_w2, d_addr, learn_addr, 'U2')
        self.U3 = init_U(self.L_b2, d_addr, learn_addr, 'U3')

        # value tapes
        # PyTorch: F.normalize(torch.randn(...), dim=...)
        def init_tape(shape, axis, learn, name):
            val = tf.random.normal(shape)
            val = tf.linalg.l2_normalize(val, axis=axis)
            return self.add_weight(
                name=name, 
                shape=val.shape, 
                dtype=tf.float32,
                initializer=tf.constant_initializer(val),
                trainable=learn
            )

        self.W1 = init_tape((self.L_w1, H, D_in), (1, 2), learn_tape_w1, 'W1')
        self.W2 = init_tape((self.L_w2, D_out, H), (1, 2), learn_tape_w2, 'W2')
        self.b2 = init_tape((self.L_b2, D_out), 1, learn_tape_b2, 'b2')

        self.act = tf.keras.activations.gelu

    @staticmethod
    def _tk(z: tf.Tensor, k: int, tau: float) -> Tuple[tf.Tensor, tf.Tensor]:
        # PyTorch: torch.topk(z, k, dim=1, largest=True, sorted=False)
        topk = tf.math.top_k(z, k=k, sorted=False)
        topv, topi = topk.values, topk.indices
        # PyTorch: torch.softmax(topv / (tau + 1e-8), dim=1)
        w = tf.nn.softmax(topv / (tau + 1e-8), axis=1)
        return topi, w

    def _address(self, x_addr: tf.Tensor):
        # PyTorch: torch.cat([self.U1, self.U2, self.U3], dim=0)
        U_pack = tf.concat([self.U1, self.U2, self.U3], axis=0)          # [Ltot, d]
        # PyTorch: x_addr @ U_pack.t()
        Z = tf.matmul(x_addr, U_pack, transpose_b=True)                 # [N, Ltot]
        s1, s2, s3 = self.L_w1, self.L_w2, self.L_b2
        # PyTorch: torch.split(Z, (s1, s2, s3), dim=1)
        z1, z2, z3 = tf.split(Z, (s1, s2, s3), axis=1)
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
        # PyTorch: N, k = topi.shape
        N = tf.shape(topi)[0]
        k = tf.shape(topi)[1]
        # PyTorch: out_dim, in_dim = W.shape[1], W.shape[2]
        out_dim = tf.shape(W)[1]
        in_dim = tf.shape(W)[2]
        
        # PyTorch: gather -> [N, k, out, in]
        # W.index_select(0, topi.reshape(-1)).view(N, k, out_dim, in_dim)
        W_sel = tf.gather(W, tf.reshape(topi, [-1]))
        W_sel = tf.reshape(W_sel, [N, k, out_dim, in_dim])
        
        # PyTorch: mix neighbors -> [N, out, in]
        # (W_sel * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        weights_exp = tf.expand_dims(tf.expand_dims(weights, -1), -1)
        W_eff = tf.reduce_sum(W_sel * weights_exp, axis=1)
        
        # PyTorch: apply -> [N, out]
        # torch.bmm(W_eff, x_flat.unsqueeze(-1)).squeeze(-1)
        x_exp = tf.expand_dims(x_flat, -1)
        res = tf.matmul(W_eff, x_exp)
        return tf.squeeze(res, -1)

    def call(self, x):
        # default values for TorchScript
        # B = 1   # dummy placeholder (not needed in TF logic)
        # T = 1
    
        # PyTorch: if x.ndim == 3
        if len(x.shape) == 3:
            # PyTorch: B, T, D = x.shape
            B, T, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
            # PyTorch: x_flat = x.reshape(B * T, D)
            x_flat = tf.reshape(x, [B * T, D])
        else:
            # PyTorch: B, T = x.shape[0], 1
            B = tf.shape(x)[0]
            T = 1
            x_flat = x
        
        # PyTorch: x_addr = self.P(x_flat)
        x_addr = self.P(x_flat)                                    # [N, d_addr]
        (i1, w1), (i2