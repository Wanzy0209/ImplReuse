num_visible = 784
num_hidden = 128
batch_size = 8
learning_rate = 0.01

input_tensor = torch.randint(0, 2, (batch_size, num_visible), dtype=torch.float32)
initial_state = torch.randint(0, 2, (5, num_visible), dtype=torch.float32)

model = BoltzmannMachine(num_visible, num_hidden, learning_rate)
for param in model.parameters():
    param.requires_grad = False

model.eval()
with torch.no_grad():
    samples_normal = model(initial_state)
    energy_normal = model.free_energy(input_tensor)
    recon_error_normal = model.contrastive_divergence(input_tensor)
    _, hidden_prob_normal = model.sample_hidden(input_tensor)
    _, visible_prob_normal = model.sample_visible(hidden_prob_normal)

compiled_model = torch.compile(model)
compiled_model.eval()
with torch.no_grad():
    samples_compiled = compiled_model(initial_state)
    energy_compiled = compiled_model.free_energy(input_tensor)
    recon_error_compiled = compiled_model.contrastive_divergence(input_tensor)
    _, hidden_prob_compiled = compiled_model.sample_hidden(input_tensor)
    _, visible_prob_compiled = compiled_model.sample_visible(hidden_prob_compiled)

diff_hidden = torch.abs(hidden_prob_normal - hidden_prob_compiled)
diff_visible = torch.abs(visible_prob_normal - visible_prob_compiled)
diff_energy = torch.abs(energy_normal - energy_compiled)
diff_samples = torch.abs(samples_normal.float() - samples_compiled.float())  # 转换为float

print(f"\n● Output Shape Verification:")
print(f"  Original samples shape: {samples_normal.shape}")
print(f"  Compiled samples shape: {samples_compiled.shape}")
print(f"  Shape consistency: {samples_normal.shape == samples_compiled.shape}")

print(f"\n● Numerical Consistency Check (rtol=0.01, atol=0.001):")
max_abs_diff_hidden = torch.max(diff_hidden).item()
max_rel_diff_hidden = torch.max(diff_hidden / (torch.abs(hidden_prob_normal) + 1e-8)).item()
mismatch_count_hidden = torch.sum(diff_hidden > 0.001).item()
total_elements_hidden = hidden_prob_normal.numel()

print(
    f"  Hidden layer - Mismatched elements: {mismatch_count_hidden} / {total_elements_hidden} ({mismatch_count_hidden / total_elements_hidden * 100:.1f}%)")
print(f"  Hidden layer - Max absolute difference: {max_abs_diff_hidden:.6f}")
print(f"  Hidden layer - Max relative difference: {max_rel_diff_hidden:.6f}")

atol_violation_hidden = max_abs_diff_hidden > 0.001
rtol_violation_hidden = max_rel_diff_hidden > 0.01

if not atol_violation_hidden and not rtol_violation_hidden:
    print(f"  ✅ Hidden layer passed consistency check")
else:
    print(f"  ❌ Hidden layer failed consistency check")
    if atol_violation_hidden:
        print(f"    Absolute tolerance violation: {max_abs_diff_hidden:.6f} > 0.001")
    if rtol_violation_hidden:
        print(f"    Relative tolerance violation: {max_rel_diff_hidden:.6f} > 0.01")

print(f"\n● Probability Distribution Differences:")
prob_max_diff_hidden = torch.max(diff_hidden).item()
prob_mean_diff_hidden = torch.mean(diff_hidden).item()
prob_l2_diff_hidden = torch.norm(diff_hidden).item()

prob_max_diff_visible = torch.max(diff_visible).item()
prob_mean_diff_visible = torch.mean(diff_visible).item()
prob_l2_diff_visible = torch.norm(diff_visible).item()

print(f"  Hidden layer - Max probability difference: {prob_max_diff_hidden:.8f}")
print(f"  Hidden layer - Mean probability difference: {prob_mean_diff_hidden:.8f}")
print(f"  Hidden layer - L2 norm difference: {prob_l2_diff_hidden:.8f}")

print(f"  Visible layer - Max probability difference: {prob_max_diff_visible:.8f}")
print(f"  Visible layer - Mean probability difference: {prob_mean_diff_visible:.8f}")
print(f"  Visible layer - L2 norm difference: {prob_l2_diff_visible:.8f}")

print(f"\n● Sample Consistency Check:")

sample_diff_count = torch.sum(samples_normal != samples_compiled).item()
sample_total = samples_normal.numel()
sample_agreement = (sample_total - sample_diff_count) / sample_total * 100
print(f"  Sample differences: {sample_diff_count} / {sample_total}")
print(f"  Sample agreement rate: {sample_agreement:.2f}%")
print(f"  Complete sample consistency: {sample_diff_count == 0}")

print(f"\n● Energy Value Comparison:")
energy_diff = torch.mean(diff_energy).item()
energy_mean_normal = torch.mean(energy_normal).item()
energy_mean_compiled = torch.mean(energy_compiled).item()
print(f"  Original energy mean: {energy_mean_normal:.6f}")
print(f"  Compiled energy mean: {energy_mean_compiled:.6f}")
print(f"  Energy difference: {energy_diff:.6f}")
print(f"  Energy consistency: {energy_diff < 0.01}")

print(f"\n● Reconstruction Error Comparison:")
recon_diff = abs(recon_error_normal - recon_error_compiled)
print(f"  Original reconstruction error: {recon_error_normal:.6f}")
print(f"  Compiled reconstruction error: {recon_error_compiled:.6f}")
print(f"  Reconstruction error difference: {recon_diff:.6f}")
print(f"  Reconstruction consistency: {recon_diff < 0.01}")

print(f"\n● Sample Value Analysis:")
print(f"  Original samples - True count: {torch.sum(samples_normal).item()}/{sample_total}")
print(f"  Compiled samples - True count: {torch.sum(samples_compiled).item()}/{sample_total}")
print(f"  Sample value agreement: {sample_agreement:.2f}%")