import torch
import torch.nn.functional as F

def test_addmm_with_lstm_like_pattern():
    """
    Test case based on Issue 167313 and the similarity to tf.keras.layers.LSTMCell.
    
    The bug occurs when torch.compile optimizes torch.addmm followed by a pointwise 
    operation (like sigmoid in LSTM cells or relu in the original bug report). 
    The optimization incorrectly ignores the alpha and beta parameters, defaulting 
    them to 1.0.
    
    This test mimics the LSTM cell pattern (MatMul -> Add -> Activation) using 
    torch.addmm to ensure the fix handles this common use case correctly.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dimensions mimicking a small LSTM cell step
    batch_size = 2
    input_size = 3
    hidden_size = 4
    
    # Inputs: x (current input) and h_prev (previous hidden state)
    x = torch.randn(batch_size, input_size, device=device)
    h_prev = torch.randn(batch_size, hidden_size, device=device)
    
    # Concatenate input and hidden state (common in LSTM cells)
    # Shape: [batch_size, input_size + hidden_size]
    xm = torch.cat([x, h_prev], dim=1)
    
    # Weights and Bias
    # Shape: [input_size + hidden_size, hidden_size]
    weights = torch.randn(input_size + hidden_size, hidden_size, device=device)
    # Shape: [hidden_size]
    bias = torch.randn(hidden_size, device=device)
    
    # Define alpha and beta that are NOT 1.0 to trigger the bug
    alpha = 0.5
    beta = 0.5

    # Function mimicking an LSTM cell gate calculation:
    # activation(beta * bias + alpha * (input @ weights))
    def lstm_gate_step(xm, weights, bias, alpha, beta):
        # The bug specifically triggers when addmm is followed by a pointwise op.
        # Here we use sigmoid, similar to the gates in tf.keras.layers.LSTMCell.
        return torch.sigmoid(torch.addmm(bias, xm, weights, alpha=alpha, beta=beta))

    # 1. Run Eager mode
    expected = lstm_gate_step(xm, weights, bias, alpha, beta)
    
    # 2. Run Compiled mode (Inductor)
    compiled_step = torch.compile(lstm_gate_step)
    actual = compiled_step(xm, weights, bias, alpha, beta)
    
    # 3. Assert correctness
    # If the bug exists, 'actual' will be calculated with alpha=1.0, beta=1.0,
    # resulting in a significant mismatch.
    if not torch.allclose(expected, actual, atol=1e-4, rtol=1e-4):
        print("Bug detected: alpha/beta parameters were ignored in compiled mode.")
        print(f"Expected (Eager):\n{expected}")
        print(f"Actual (Compiled):\n{actual}")
        raise AssertionError("Output mismatch between eager and compiled modes.")
    else:
        print("Test passed: alpha/beta parameters were respected.")

if __name__ == "__main__":
    test_addmm_with_lstm_like_pattern()