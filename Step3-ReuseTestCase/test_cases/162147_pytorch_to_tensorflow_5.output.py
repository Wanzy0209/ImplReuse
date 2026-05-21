import tensorflow as tf
import numpy as np
import math

class FastLearnedCellX3(tf.keras.Model):
    def __init__(self, D_in, H, D_out,
                 L_w1=12, L_w2=12, L_b2=12,
                 k1=3, k2=3, k3=3, tau1=1.0, tau2=1.0, tau3=1.0,
                 d_addr=64,           # address bottleneck
                 learn_addr=False,
                 learn_tape_w1=True, learn_tape_w2=True, learn_tape_b2=True):
        super().__init__()
        self.D_in, self.H, self.D_out = D_in, H, D_out

        # keep sizes as plain Python ints
        self.L_w1, self.L_w2, self.L_b2 = int(L_w1), int(L_w2), int(L_b2)
        self.k1, self.k2, self.k3 = int(k1), int(k2), int(k3)
        self.t1, self.t2, self.t3 = float(tau1), float(tau2), float(tau3)

        # address bottleneck
        self.P = tf.keras.layers.Dense(d_addr, use_bias=False)
        if not learn_addr:
            self.P.trainable = False
            # Initialize weights manually to match PyTorch logic
            self.P.build((None, D_in))
            w_init = self.P.get_weights()
            new_w = np.random.normal(0, 1.0/math.sqrt(D_in), w_init[0].shape).astype(np.float32)
            self.P.set_weights([new_w])

        def init_U(L, d, learn):
            # Initialize U with normal distribution, subtract mean, normalize
            U_val = np.random.randn(L, d).astype(np.float32)
            U_val = U_val - U_val.mean(axis=1, keepdims=True)
            norm = np.linalg.norm(U_val, axis=1, keepdims=True)
            U_val = U_val / (norm + 1e-8)
            return tf.Variable(U_val, trainable=learn, dtype=tf.float32)

        # three unembeddings in addr-space
        self.U1 = init_U(self.L_w1, d_addr, learn_addr)
        self.U2 = init_U(self.L_w2, d_addr, learn_addr)
        self.U3 = init_U(self
    assert L_w1
