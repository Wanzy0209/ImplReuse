================================================================================
SDPA NaN Gradient Test Case
================================================================================

Loading debug file: sdpa_nan_debug_Transformer_res-6.pt
Debug info from module: Transformer_res-6
NaN detected in: grad_q

=== Tensor Shapes ===
q: torch.Size([20, 4, 4, 32])
k: torch.Size([20, 4, 4, 32])
v: torch.Size([20, 4, 4, 32])
attn_output: torch.Size([20, 4, 4, 32])

=== Forward Pass Statistics ===
q - min: -9.187500e+00, max: 1.031250e+01, mean: 3.564453e-02, std: 3.281250e+00
q - has NaN: False, has Inf: False, NaN count: 0
k - min: -1.037500e+01, max: 9.625000e+00, mean: 3.247070e-02, std: 3.140625e+00
k - has NaN: False, has Inf: False, NaN count: 0
v - min: -3.417969e-01, max: 3.300781e-01, mean: 4.577637e-03, std: 1.064453e-01
v - has NaN: False, has Inf: False, NaN count: 0
attn_output - min: -2.158203e-01, max: 1.806641e-01, mean: 7.385254e-03, std: 7.275391e-02
attn_output - has NaN: False, has Inf: False, NaN count: 0

=== Gradient Statistics ===
grad_attn_output - min: -8.583069e-06, max: 8.165836e-06, mean: -7.566996e-09, std: 1.601875e-06
grad_attn_output - has NaN: False, has Inf: False, NaN count: 0
grad_q - min: -1.427907e-10, max: 1.518856e-10, mean: 1.803002e-13, std: 1.057288e-11
grad_q - has NaN: True, has Inf: False, NaN count: 640
grad_k - min: -1.182343e-10, max: 1.145963e-10, mean: -1.945111e-13, std: 8.810730e-12
grad_k - has NaN: False, has Inf: False, NaN count: 0
grad_v - min: -1.126528e-05, max: 1.096725e-05, mean: -7.566996e-09, std: 1.400709e-06
grad_v - has NaN: False, has Inf: False, NaN count: 0

=== Reconstructing Forward Pass ===

=== Manual Attention Score Calculation ===
Attention scores shape: torch.Size([20, 4, 4, 4])
attn_scores (Q@K^T / sqrt(d)) - min: -1.310000e+02, max: 2.737500e+01, mean: -2.475000e+01, std: 4.475000e+01
attn_scores (Q@K^T / sqrt(d)) - has NaN: False, has Inf: False, NaN count: 0
Attention weights shape: torch.Size([20, 4, 4, 4])
attn_weights (softmax) - min: 5.816114e-24, max: 1.000000e+00, mean: 2.500000e-01, std: 4.335938e-01
attn_weights (softmax) - has NaN: False, has Inf: False, NaN count: 0
Manual attention output shape: torch.Size([20, 4, 4, 32])
attn_output_manual - min: -2.158203e-01, max: 1.806641e-01, mean: 7.385254e-03, std: 7.275391e-02
attn_output_manual - has NaN: False, has Inf: False, NaN count: 0

=== SDPA Forward Pass ===
qkv shape torch.Size([20, 4, 4, 32]) torch.Size([20, 4, 4, 32]) torch.Size([20, 4, 4, 32])
qkv strides (512, 32, 128, 1) (512, 32, 128, 1) (512, 32, 128, 1)
Reconstructed attn_output - has NaN: False
Reconstructed attn_output - min: -2.158203e-01, max: 1.806641e-01, mean: 7.385254e-03, std: 7.275391e-02
Reconstructed attn_output - has NaN: False, has Inf: False, NaN count: 0
attn_output difference (ignoring NaN) - max: 0.000000e+00, mean: 0.000000e+00

=== Reconstructing Backward Pass ===
Reconstructed grad_q - has NaN: True
Reconstructed grad_k - has NaN: False
Reconstructed grad_v - has NaN: False

grad_q difference (ignoring NaN) - max: 0.000000e+00, mean: 0.000000e+00
grad_q difference - NaN count: 640
grad_k difference (ignoring NaN) - max: 0.000000e+00, mean: 0.000000e+00
grad_v difference (ignoring NaN) - max: 0.000000e+00, mean: 0.000000e+00

=== Test Complete ===
You can now inspect the tensors in debug_data for further analysis.

================================================================================

Running minimal reproduction test...
================================================================================
Loading debug file: sdpa_nan_debug_Transformer_res-6.pt
Forward output has NaN: False
grad_q has NaN: True
grad_k has NaN: False
grad_v has NaN: False

✓ Successfully reproduced NaN gradient!