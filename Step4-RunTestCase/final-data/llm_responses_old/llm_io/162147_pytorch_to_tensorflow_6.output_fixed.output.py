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
        self.P = tf.keras.layers.Dense(d_addr, use_bias=False, name="P")
        if not learn_addr:
            self.P.trainable = False
            # Initialize weights manually if not learning
            # Note: Keras layers build weights on first call or explicit build
            self.P.build((None, D_in))
            # FIX: Dense layer kernel shape is (input_dim, output_dim), i.e., (D_in, d_addr)
            self.P.set_weights([tf.random.normal((D_in, d_addr), stddev=1.0/math.sqrt(D_in))])

        def init_U(L, d, learn):
            U = tf.random.normal((L, d))
            U = U - tf.reduce_mean(U, axis=1, keepdims=True)
            U = U / (tf.norm(U, axis=1, keepdims=True) + 1e-8)
            return tf.Variable(U, trainable=learn, name=f"U_{L}")

        # three unembeddings in addr-space
        self.U1 = init_U(self.L_w1, d_addr, learn_addr)
        self.U2 = init_U(self.L_w2, d_addr, learn_addr)
        self.U3 = init_U(self.L_b2, d_addr, learn_addr)

        # value tapes
        # Normalizing along dims (1,2) for W1, (1,2) for W2, 1 for b2
        w1_init = tf.random.normal((self.L_w1, H, D_in))
        self.W1 = tf.Variable(tf.nn.l2_normalize(w1_init, axis=(1,2)), trainable=learn_tape_w1, name="W1")
        
        w2_init = tf.random.normal((self.L_w2, D_out, H))
        self.W2 = tf.Variable(tf.nn.l2_normalize(w2_init, axis=(1,2)), trainable=learn_tape_w2, name="W2")
        
        b2_init = tf.random.normal((self.L_b2, D_out))
        self.b2 = tf.Variable(tf.nn.l2_normalize(b2_init, axis=1), trainable=learn_tape_b2, name="b2")

    @staticmethod
    def _tk(z, k, tau):
        # tf.math.top_k returns values and indices
        topk_values, topk_indices = tf.math.top_k(z, k=k, sorted=False)
        w = tf.nn.softmax(topk_values / (tau + 1e-8), axis=1)
        return topk_indices, w

    def _address(self, x_addr):
        # fused logits for all three heads
        # U_pack shape: [Ltot, d]
        U_pack = tf.concat([self.U1, self.U2, self.U3], axis=0)
        # Z shape: [N, Ltot]
        Z = tf.matmul(x_addr, U_pack, transpose_b=True)
        
        s1, s2, s3 = self.L_w1, self.L_w2, self.L_b2
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
        
        # Gather W based on indices topi
        # W_selected shape: [N, k, out, in]
        W_selected = tf.gather(W, topi)
        
        # Expand x_flat to [N, 1, in]
        x_exp = tf.expand_dims(x_flat, 1)
        
        # Multiply: [N, k, out, in] * [N, 1, in] -> [N, k, out, in]
        # Then sum over last dim (in) -> [N, k, out]
        weighted_W = tf.reduce_sum(W_selected * x_exp, axis=-1)
        
        # Apply weights: [N, k, out] * [N, k, 1] -> sum over k -> [N, out]
        weights_exp = tf.expand_dims(weights, -1)
        result = tf.reduce_sum(weighted_W * weights_exp, axis=1)
        
        return result

    @tf.function
    def call(self, x, training=False):
        # Apply the similar API: tf.compat.v1.nn.dropout
        # This adapts the original test case to include the requested API behavior
        # We apply it to the input to test its interaction with the complex graph.
        if training:
            # Using a fixed rate for testing
            x = tf.compat.v1.nn.dropout(x, rate=0.1)
        
        # Address bottleneck
        x_addr = self.P(x)
        
        # Get addresses
        (i1, w1), (i2, w2), (i3, w3) = self._address(x_addr)
        
        # Apply mixtures
        # x is [N, D_in]
        # W1 is [L_w1, H, D_in] -> output [N, H]
        h = self._apply_mixture(x, i1, w1, self.W1)
        h = tf.nn.gelu(h)
        
        # W2 is [L_w2, D_out, H] -> output [N, D_out]
        # h is [N, H]
        y = self._apply_mixture(h, i2, w2, self.W2)
        
        # b2 is [L_b2, D_out] -> output [N, D_out]
        # We need to broadcast a dummy input for bias or handle it differently.
        # The original code implies b2 is mixed similarly. 
        # Let's assume a dummy input of ones or reuse h if dimensions match logic.
        # Based on typical MLPs, bias is added. Here it seems to be a "learned bias tape".
        # Let's use a tensor of ones with shape [N, 1] to mix the bias.
        ones = tf.ones((tf.shape(x)[0], 1), dtype=x.dtype)
        b_out = self._apply_mixture(ones, i3, w3, self.b2)
        
        return y + b_out

def test_fast_learned_cell_tf():
    # Parameters
    D_in, H, D_out = 128, 256, 10
    batch_size = 32
    
    # Initialize Model
    model = FastLearnedCellX3(D_in, H, D_out)
    
    # Create dummy input
    x = tf.random.normal((batch_size, D_in))
    
    # Run forward pass in training mode (activates dropout)
    # This tests the graph complexity and the specific API integration
    with tf.GradientTape() as tape:
        output = model(x, training=True)
        loss = tf.reduce_mean(output**2)
    
    # Check output shape
    assert output.shape == (batch_size, D_out), f"Output shape mismatch: {output.shape}"
    
    # Calculate gradients (to test backprop through the complex graph)
    grads = tape.gradient(loss, model.trainable_variables)
    
    # Verify gradients are computed (not None)
    assert grads is not None
    assert all(g is not None for g in grads), "Some gradients are None"
    
    print("TensorFlow test passed. Graph execution and dropout integration verified.")

if __name__ == "__main__":
    test_fast_learned_cell_tf()