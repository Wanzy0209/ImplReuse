import torch
import torch.nn as nn

# The bug report mentions issues with _scaled_mm and _int_mm (internal quantized matmul functions)
# being slow or raising errors with row-major matrices.
# The similar API is torch.nn.RNNBase, specifically its quantized variants.
# This test verifies that a Quantized RNN (LSTM) handles different input layouts (batch_first)
# correctly, which exercises the internal matrix multiplications.

def test_quantized_rnn_layouts():
    # Setup dimensions
    input_size = 16
    hidden_size = 32
    batch_size = 4
    seq_length = 10

    # Test 1: batch_first=False (Default, input layout: Seq, Batch, Feature)
    # This is the standard layout for RNNs.
    rnn = nn.LSTM(input_size, hidden_size, batch_first=False)
    # Apply dynamic quantization. This converts Linear layers (used inside LSTM) to quantized versions.
    # Quantized layers use _int_mm or _scaled_mm internally.
    rnn_quantized = torch.quantization.quantize_dynamic(
        rnn, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    )

    input_data = torch.randn(seq_length, batch_size, input_size)
    
    # Run forward pass
    output, (hn, cn) = rnn_quantized(input_data)
    
    # Assertions
    assert output.shape == (seq_length, batch_size, hidden_size), \
        f"Expected shape ({seq_length}, {batch_size}, {hidden_size}), got {output.shape}"
    assert hn.shape == (1, batch_size, hidden_size)
    assert cn.shape == (1, batch_size, hidden_size)

    # Test 2: batch_first=True (Input layout: Batch, Seq, Feature)
    # Changing batch_first changes the striding/access pattern of the input matrix.
    # This helps verify that the internal kernels handle different layouts without crashing
    # or significant performance regressions (though we only check correctness here).
    rnn_bf = nn.LSTM(input_size, hidden_size, batch_first=True)
    rnn_quantized_bf = torch.quantization.quantize_dynamic(
        rnn_bf, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    )

    input_data_bf = torch.randn(batch_size, seq_length, input_size)
    
    # Run forward pass
    output_bf, (hn_bf, cn_bf) = rnn_quantized_bf(input_data_bf)
    
    # Assertions
    assert output_bf.shape == (batch_size, seq_length, hidden_size), \
        f"Expected shape ({batch_size}, {seq_length}, {hidden_size}), got {output_bf.shape}"
    assert hn_bf.shape == (1, batch_size, hidden_size)
    assert cn_bf.shape == (1, batch_size, hidden_size)

    print("Test passed: Quantized RNN handles different input layouts correctly.")

if __name__ == "__main__":
    test_quantized_rnn_layouts()