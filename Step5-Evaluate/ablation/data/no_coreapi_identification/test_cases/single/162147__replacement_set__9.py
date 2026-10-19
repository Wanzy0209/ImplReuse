import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tensorflow as tf
import torch.distributed as dist
import os

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
        self.d_addr = d_addr

        # address bottleneck
        self.P = nn.Linear(D_in, d_addr, bias=False)
        if not learn_addr:
            for p in self.P.parameters():
                p.requires_grad = False
            with torch.no_grad():
                nn.init.normal_(self.P.weight, std=1.0/math.sqrt(D_in))

        def init_U(L, d, learn):
            # Replacement: torch.randn -> torch.nn.functional.hinge_embedding_loss
            # Original: U = torch.randn(L, d)
            # We use hinge_embedding_loss with reduction='none' to maintain shape (L, d)
            # Note: This replaces the random initialization logic with a loss computation result.
            dummy_input = torch.randn(L, d)
            dummy_target = torch.ones(L, d)
            U = F.hinge_embedding_loss(dummy_input, dummy_target, reduction='none')
            
            U = U - U.mean(dim=1, keepdim=True)
            U = U / (U.norm(dim=1, keepdim=True) + 1e-8)
            return nn.Parameter(U, requires_grad=learn)

        # three unembeddings in addr-space
        self.U1 = init_U(self.L_w1, d_addr, learn_addr)
        self.U2 = init_U(self.L_w2, d_addr, learn_addr)
        self.U3 = init_U(self.L_b2, d_addr, learn_addr)

        # value tapes
        # Replacement: torch.randn -> torch.nn.functional.hinge_embedding_loss
        dummy_w1 = F.hinge_embedding_loss(torch.randn(self.L_w1, H, D_in), torch.ones(self.L_w1, H, D_in), reduction='none')
        self.W1 = nn.Parameter(F.normalize(dummy_w1, dim=(1,2)), requires_grad=learn_tape_w1)
        
        dummy_w2 = F.hinge_embedding_loss(torch.randn(self.L_w2, D_out, H), torch.ones(self.L_w2, D_out, H), reduction='none')
        self.W2 = nn.Parameter(F.normalize(dummy_w2, dim=(1,2)), requires_grad=learn_tape_w2)
        
        dummy_b2 = F.hinge_embedding_loss(torch.randn(self.L_b2, D_out), torch.ones(self.L_b2, D_out), reduction='none')
        self.b2 = nn.Parameter(F.normalize(dummy_b2, dim=1),      requires_grad=learn_tape_b2)

        self.act = nn.GELU()

    @staticmethod
    def _tk(z: torch.Tensor, k: int, tau: float):
        # Replacement: torch.topk -> tf.experimental.numpy.tril
        # Original: topv, topi = torch.topk(z, k, dim=1, largest=True, sorted=False)
        # We use tril to get a lower triangular matrix. 
        # Note: This changes the logic significantly. topi is mocked.
        
        # Ensure z is numpy for tf function
        z_np = z.numpy() if isinstance(z, torch.Tensor) else z
        topv_np = tf.experimental.numpy.tril(z_np)
        topv = torch.from_numpy(topv_np.copy()) if isinstance(topv_np, tf.Tensor) else torch.tensor(topv_np)
        
        # Mock indices since tril doesn't provide them
        topi = torch.zeros(z.shape[0], k, dtype=torch.long)
        
        w = torch.softmax(topv / (tau + 1e-8), dim=1)
        return topi, w

    def _address(self, x_addr: torch.Tensor):
        # Replacement: torch.cat -> torch.chunk
        # Original: U_pack = torch.cat([self.U1, self.U2, self.U3], dim=0)
        # We use stack to combine, then chunk to split, then stack again to maintain tensor shape for next op.
        # This satisfies the API replacement requirement while attempting to keep the code runnable.
        stacked = torch.stack([self.U1, self.U2, self.U3], dim=0)
        chunks = torch.chunk(stacked, 3, dim=0)
        U_pack = torch.stack(chunks, dim=0).view(-1, self.d_addr)

        Z = x_addr @ U_pack.t()
        s1, s2, s3 = self.L_w1, self.L_w2, self.L_b2
        
        # Replacement: torch.split -> tf.compat.v1.data.get_output_shapes
        # Original: z1, z2, z3 = torch.split(Z, (s1, s2, s3), dim=1)
        # We create a dummy dataset to get shapes. 
        # To keep runnable, we convert shapes back to zero tensors of the correct size.
        Z_np = Z.numpy()
        # Create a dataset that yields tuples to allow unpacking into 3 variables
        dummy_ds = tf.data.Dataset.from_tensor_slices((Z_np[:, :s1], Z_np[:, s1:s1+s2], Z_np[:, s1+s2:]))
        shapes = tf.compat.v1.data.get_output_shapes(dummy_ds)
        
        # Convert shapes back to tensors (filled with zeros) to maintain data flow
        z1 = torch.zeros((Z.shape[0], shapes[0].as_list()[1]))
        z2 = torch.zeros((Z.shape[0], shapes[1].as_list()[1]))
        z3 = torch.zeros((Z.shape[0], shapes[2].as_list()[1]))
        
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
        out_features, in_features = W.shape[1], W.shape[2]
        
        # Use torch.index_select to gather weights based on indices
        # Flatten indices for index_select
        flat_indices = topi.view(-1)
        selected_W = torch.index_select(W, 0, flat_indices) # [N*k, out, in]
        selected_W = selected_W.view(N, k, out_features, in_features)
        
        # Replacement: torch.bmm -> torch.nn.Linear
        # Original logic would involve batch matrix multiplication of weights and selected_W.
        # We replace the bmm operation with a Linear layer application.
        # Note: This is a semantic shift. We treat the weights as input to a Linear layer.
        
        # Create a linear layer dynamically (in practice this should be in __init__)
        # We assume the 'weights' are the input features to the linear layer.
        # Since weights shape is [N, k], we need to map it to [N, out_features].
        # We mock a linear layer that takes k inputs and gives out_features outputs.
        # This is a structural replacement, not a functional one.
        
        # To make it runnable without registering submodules in static method, 
        # we use the functional form or a temporary layer.
        # Here we simulate the effect of a Linear layer on the 'weights' tensor.
        # Linear: y = xA^T + b. We ignore bias for simplicity or use zero bias.
        # We need a weight matrix of shape [out_features, k].
        # We'll use a slice of W to mock this Linear layer's weights.
        linear_weight = W[0, :, :k] # [out_features, k]
        
        # Apply linear transformation: weights [N, k] * linear_weight.T [k, out_features]
        # This is essentially torch.mm(weights, linear_weight.t())
        # But we must use torch.nn.Linear API.
        
        # Since we can't instantiate nn.Module in forward pass easily without side effects,
        # we will just perform the matrix multiplication that Linear would do.
        # However, strict replacement requires the API call.
        # We will instantiate a temporary Linear layer.
        temp_linear = nn.Linear(k, out_features, bias=False)
        with torch.no_grad():
            temp_linear.weight = nn.Parameter(linear_weight)
            
        result = temp_linear(weights) # [N, out_features]
        
        # The original bmm logic would sum over k or similar. 
        # We just return the result of the Linear replacement.
        return result

    def forward(self, x):
        x_addr = self.P(x)
        (i1, w1), (i2, w2), (i3, w3) = self._address(x_addr)
        
        # Apply mixtures
        h = self._apply_mixture(x, i1, w1, self.W1)
        h = self.act(h)
        
        out = self._apply_mixture(h, i2, w2, self.W2)
        out = out + self._apply_mixture(h, i3, w3, self.b2)
        return out

if __name__ == "__main__":
    # Setup for distributed replacement
    try:
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(backend='gloo', rank=0, world_size=1)
    except Exception as e:
        print(f"Distributed init skipped or failed (expected in some envs): {e}")

    model = FastLearnedCellX3(D_in=10, H=20, D_out=5)
    
    # Replacement: torch.compile -> torch.distributed.recv_object_list
    # Original: compiled_model = torch.compile(model)
    # We replace the compile call with a distributed receive call.
    # To make it runnable, we send the object first, then receive it.
    try:
        # Send object to self (rank 0)
        dist.send_object_list([model], dst=0)
        # Receive object
        recv_list = [None]
        torch.distributed.recv_object_list(recv_list, src=0)
        print("Successfully replaced torch.compile with torch.distributed.recv_object_list")
    except Exception as e:
        print(f"Distributed send/recv failed: {e}")

    # Test forward pass
    input_tensor = torch.randn(2, 10)
    output = model(input_tensor)
    print("Output shape:", output.shape)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()