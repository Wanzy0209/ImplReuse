import torch

def test_dtype_alpha_combination(dtypeA, dtypeB, alpha):
    x = torch.ones([1], dtype=dtypeA, device="mps")
    y = torch.ones([1], dtype=dtypeB, device="mps")
    try:
        torch.add(x, y, alpha=alpha)
        print(f"{dtypeA}, {dtypeB}, alpha={alpha} succeeds")
    except RuntimeError as e:
        print(f"{dtypeA}, {dtypeB}, alpha={alpha} fails:", e)

test_dtype_alpha_combination(torch.int, torch.int, 1) # succeeds
test_dtype_alpha_combination(torch.int, torch.int, 2) # succeeds

test_dtype_alpha_combination(torch.int, torch.float, 1) # succeeds
test_dtype_alpha_combination(torch.int, torch.float, 2) # fails: Failed to create function state object for: add_alpha_dense_cast_float_int

test_dtype_alpha_combination(torch.int8, torch.uint8, 1) # succeeds
test_dtype_alpha_combination(torch.int8, torch.uint8, 2) # fails: Failed to create function state object for: add_alpha_dense_cast_short_char

test_dtype_alpha_combination(torch.bool, torch.int, 1) # succeeds
test_dtype_alpha_combination(torch.bool, torch.int, 2) # fails: Failed to create function state object for: add_alpha_dense_cast_int_bool

test_dtype_alpha_combination(torch.bool, torch.float, 1) # succeeds
test_dtype_alpha_combination(torch.bool, torch.float, 2) # fails: Failed to create function state object for: add_alpha_dense_cast_float_bool

# Similar failures for sub.Tensor and also for scalar variants: add.Scalar, sub.Scalar, rsub.Scalar
def test_op_tensor_scalar_combination(op, dtype, scalar, alpha):
    x = torch.ones([1], dtype=dtype, device="mps")
    try:
        op(x, scalar, alpha=alpha)
        print(f"{op.__name__}, {dtype}, {scalar}, alpha={alpha} succeeds")
    except RuntimeError as e:
        print(f"{op.__name__}, {dtype}, {scalar}, alpha={alpha} fails:", e)

# add.Scalar
test_op_tensor_scalar_combination(torch.add, torch.int, 1, 1) # succeeds
test_op_tensor_scalar_combination(torch.add, torch.int, 1, 2) # succeeds
test_op_tensor_scalar_combination(torch.add, torch.int, 1.0, 1) # succeeds
test_op_tensor_scalar_combination(torch.add, torch.int, 1.0, 2) # fails: Failed to create function state object for: add_alpha_dense_cast_float_int

# sub.Scalar
test_op_tensor_scalar_combination(torch.sub, torch.int, 1, 1) # succeeds
test_op_tensor_scalar_combination(torch.sub, torch.int, 1, 2) # succeeds
test_op_tensor_scalar_combination(torch.sub, torch.int, 1.0, 1) # succeeds
test_op_tensor_scalar_combination(torch.sub, torch.int, 1.0, 2) # fails: Failed to create function state object for: sub_alpha_dense_cast_float_int

# rsub.Scalar
test_op_tensor_scalar_combination(torch.rsub, torch.int, 1, 1) # succeeds
test_op_tensor_scalar_combination(torch.rsub, torch.int, 1, 2) # fails: Failed to create function state object for: sub_alpha_dense_cast_int_long
test_op_tensor_scalar_combination(torch.rsub, torch.int, 1.0, 1) # succeeds
test_op_tensor_scalar_combination(torch.rsub, torch.int, 1.0, 2) # fails: Undefined type Double