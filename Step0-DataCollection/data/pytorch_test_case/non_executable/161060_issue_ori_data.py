_to_copy_default_17 = torch.ops.aten._to_copy.default(detach_default_8, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
        unsqueeze_default_86 = torch.ops.aten.unsqueeze.default(_to_copy_default_17, 1);  _to_copy_default_17 = None
        unsqueeze_default_87 = torch.ops.aten.unsqueeze.default(unsqueeze_default_86, -1);  unsqueeze_default_86 = None
        _tensor_constant12 = self._tensor_constant12
        mul_tensor_5 = torch.ops.aten.mul.Tensor(unsqueeze_default_87, _tensor_constant12);  unsqueeze_default_87 = _tensor_constant12 = None
        cos_default_1 = torch.ops.aten.cos.default(mul_tensor_5)
        sin_default_1 = torch.ops.aten.sin.default(mul_tensor_5);  mul_tensor_5 = None
        split_tensor_1 = torch.ops.aten.split.Tensor(transpose_int_1, 64, -1);  transpose_int_1 = None
        getitem_6 = split_tensor_1[0]
        getitem_7 = split_tensor_1[1];  split_tensor_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(getitem_6, cos_default_1)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(getitem_7, sin_default_1)
        return mul_tensor_7