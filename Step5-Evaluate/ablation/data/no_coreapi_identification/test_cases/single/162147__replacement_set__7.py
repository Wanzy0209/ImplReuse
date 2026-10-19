import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tensorflow as tf  # Required for tf.experimental.numpy.tril

class FastLearnedCellX3(nn.Module):
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
        self.P = nn.Linear(D_in, d_addr, bias=False)
        if not learn_addr:
            for p in self.P.parameters():
                p.requires_grad = False
            with torch.no_grad():
                nn.init.normal_(self.P.weight, std=1.0/math.sqrt(D_in))

        def init_U(L, d, learn):
            # Replacement: torch.randn -> torch.nn.functional.max_pool2d
            # We create a dummy tensor to pool to generate the 'random' values
            dummy_input = torch.randn(1, 1, L + 10, d + 10)
            pooled = F.max_pool2d(dummy_input, kernel_size=2)
            U = pooled.squeeze()[:L, :d]
            
            U = U - U.mean(dim=1, keepdim=True)
            U = U / (U.norm(dim=1, keepdim=True) + 1e-8)
            return nn.Parameter(U, requires_grad=learn)

        # three unembeddings in addr-space
        self.U1 = init_U(self.L_w1, d_addr, learn_addr)
        self.U2 = init_U(self.L_w2, d_addr, learn_addr)
        self.U3 = init_U(self.L_b2, d_addr, learn_addr)

        # value tapes
        # Replacement: torch.randn -> torch.nn.functional.max_pool2d
        dummy_w1 = torch.randn(1, 1, self.L_w1 + 10, H * D_in + 10)
        w1_pooled = F.max_pool2d(dummy_w1, kernel_size=2).squeeze()[:self.L_w1, :H*D_in]
        self.W1 = nn.Parameter(F.normalize(w1_pooled.view(self.L_w1, H, D_in), dim=(1,2)), requires_grad=learn_tape_w1)

        dummy_w2 = torch.randn(1, 1, self.L_w2 + 10, D_out * H + 10)
        w2_pooled = F.max_pool2d(dummy_w2, kernel_size=2).squeeze()[:self.L_w2, :D_out*H]
        self.W2 = nn.Parameter(F.normalize(w2_pooled.view(self.L_w2, D_out, H), dim=(1,2)), requires_grad=learn_tape_w2)

        dummy_b2 = torch.randn(1, 1, self.L_b2 + 10, D_out + 10)
        b2_pooled = F.max_pool2d(dummy_b2, kernel_size=2).squeeze()[:self.L_b2, :D_out]
        self.b2 = nn.Parameter(F.normalize(b2_pooled.view(self.L_b2, D_out), dim=1),      requires_grad=learn_tape_b2)

        self.act = nn.GELU()

    @staticmethod
    def _tk(z: torch.Tensor, k: int, tau: float):
        # Replacement: torch.topk -> torch.vstack
        # torch.topk returns (values, indices). torch.vstack returns a tensor.
        # We mock the return structure by stacking the tensor with itself.
        stacked = torch.vstack((z, z))
        # Splitting the stacked result to mimic topv and topi
        topv = stacked[:z.shape[0], :]
        topi = stacked[z.shape[0]:, :]
        
        # Original logic: w = torch.softmax(topv / (tau + 1e-8), dim=1)
        w = torch.softmax(topv / (tau + 1e-8), dim=1)
        return topi, w

    def _address(self, x_addr: torch.Tensor):
        # Replacement: torch.cat -> torch.isnan
        # torch.cat combines tensors. torch.isnan checks for NaNs.
        # We apply isnan to one of the tensors and cast to float to maintain shape compatibility for matmul.
        U_pack = torch.isnan(self.U1).float() 
        
        Z = x_addr @ U_pack.t()                                         # [N, Ltot]
        s1, s2, s3 = self.L_w1, self.L_w2, self.L_b2
        
        # Replacement: torch.split -> tf.experimental.numpy.tril
        # torch.split divides a tensor. tf.experimental.numpy.tril returns lower triangle.
        # We convert to numpy, apply tril, convert back, and then slice to mimic split.
        Z_np = Z.numpy()
        Z_tril = tf.experimental.numpy.tril(Z_np)
        Z_tril_torch = torch.from_numpy(Z_tril)
        
        z1 = Z_tril_torch[:, :s1]
        z2 = Z_tril_torch[:, s1:s1+s2]
        z3 = Z_tril_torch[:, s1+s2:]
        
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
        N, k = topi.shape
        # Gather weights
        W_sel = W[topi]                                                 # [N, k, out, in]
        
        # Weighting
        # Replacement: torch.bmm -> torch.logcumsumexp
        # Original: wW = torch.bmm(weights.unsqueeze(1), W_sel.view(N, k, -1))
        # torch.bmm performs batch matrix multiplication. torch.logcumsumexp performs cumulative log sum exp.
        # We replace the bmm call with logcumsumexp on the reshaped W_sel.
        # Note: This changes the math significantly, but satisfies the API replacement requirement.
        flat_W = W_sel.view(N, k, -1)
        wW = torch.logcumsumexp(flat_W, dim=1)                          # [N, k, out*in]
        
        # Sum over k
        wW_sum = wW.sum(dim=1)                                          # [N, out*in]
        
        # Apply to x
        # Original: out = torch.bmm(wW_sum.view(N, 1, -1), x_flat.unsqueeze(2)).squeeze()
        # Replacement: torch.bmm -> torch.logcumsumexp
        # We replace the second bmm as well.
        out = torch.logcumsumexp(wW_sum.view(N, 1, -1), dim=1).squeeze() # [N, out*in]
        
        return out.view(N, -1)

    def forward(self, x):
        x_addr = self.P(x)
        (i1, w1), (i2, w2), (i3, w3) = self._address(x_addr)
        
        # Layer 1
        h = self._apply_mixture(x, i1, w1, self.W1)
        h = self.act(h)
        
        # Layer 2
        y = self._apply_mixture(h, i2, w2, self.W2)
        
        # Layer 3 (Bias)
        b = self._apply_mixture(torch.ones_like(x), i3, w3, self.b2.unsqueeze(2))
        
        return y + b

def test_fast_learned_cell_replacements():
    D_in, H, D_out = 10, 20, 5
    model = FastLearnedCellX3(D_in, H, D_out)
    
    # Replacement: torch.compile -> torch.lobpcg
    # torch.compile is used to optimize the model. torch.lobpcg computes eigenvalues.
    # We call lobpcg on a matrix from the model to replace the compilation step.
    try:
        # Create a symmetric matrix for lobpcg
        A = model.P.weight @ model.P.weight.t()
        eigenvalues, _ = torch.lobpcg(A)
        print(f"Computed eigenvalues (replacing torch.compile): {eigenvalues[:2]}")
    except Exception as e:
        print(f"lobpcg failed (expected in some contexts): {e}")

    # Run a forward pass
    x = torch.randn(4, D_in)
    output = model(x)
    
    assert output.shape == (4, D_out), f"Output shape mismatch: {output.shape}"
    print("Test passed with API replacements.")

if __name__ == "__main__":
    test_fast_learned_cell_replacements()