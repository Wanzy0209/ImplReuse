import torch
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.manual_seed(52676)
def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, sentinel):
    var_node_6 = arg_0
    var_node_7 = arg_1
    var_node_5 = torch.matmul(var_node_6.to(torch.float64), var_node_7.to(torch.float64))
    var_node_9 = torch.full((9, 11, 12), 1.5758497316910556, dtype=torch.float64)
    var_node_10 = arg_2
    var_node_8 = torch.matmul(var_node_9.to(torch.float64), var_node_10.to(torch.float64))
    var_node_4 = torch.matmul(var_node_5.to(torch.float64), var_node_8.to(torch.float64))
    var_node_13 = arg_3
    var_node_14 = arg_4
    var_node_12 = torch.matmul(var_node_13.to(torch.float64), var_node_14.to(torch.float64))
    var_node_15 = arg_5
    var_node_11 = torch.matmul(var_node_12.to(torch.float64), var_node_15.to(torch.float64))
    var_node_3 = torch.matmul(var_node_4.to(torch.float64), var_node_11.to(torch.float64))
    var_node_17 = arg_6
    var_node_18 = arg_7
    var_node_16 = torch.matmul(var_node_17.to(torch.float64), var_node_18.to(torch.float64))
    var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_16.to(torch.float64))
    var_node_23 = torch.full((156, 8), -0.5249394453404403, dtype=torch.float64)
    var_node_24 = torch.full((8, 9), 0.9331226188585692, dtype=torch.float64)
    var_node_22 = torch.matmul(var_node_23.to(torch.float64), var_node_24.to(torch.float64))
    var_node_26 = torch.full((9, 13), -0.9276381954691514, dtype=torch.float64)
    var_node_27 = torch.full((13, 16), 0.024752238943232543, dtype=torch.float64)
    var_node_25 = torch.matmul(var_node_26.to(torch.float64), var_node_27.to(torch.float64))
    var_node_21 = torch.matmul(var_node_22.to(torch.float64), var_node_25.to(torch.float64))
    var_node_29 = arg_8
    _x_nz = torch.zeros((9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=torch.bool, device=var_node_29.device)
    _x_nz_flat = _x_nz.reshape(-1)
    _x_nz_flat[:9] = True
    var_node_28 = torch.nonzero(_x_nz)
    var_node_20 = torch.nn.functional.embedding(torch.clamp(var_node_28.to(torch.int64), 0, var_node_21.size(0)-1), var_node_21)
    return var_node_20