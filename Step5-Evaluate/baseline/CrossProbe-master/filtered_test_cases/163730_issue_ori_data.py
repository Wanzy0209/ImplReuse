def aten_bilinear(input1, input2, weight, bias=None):
    # input1: (*, H_in1), input2: (*, H_in2), weight: (H_out, H_in1, H_in2)
    # Compute x1^T A x2 for each output feature
    output = torch.einsum('...i,...j,oij->...o', input1, input2, weight)
    if bias is not None:
        output += bias
    return output