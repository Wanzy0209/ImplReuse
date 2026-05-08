>>> a = torch.randn(3)
>>> b = torch.randn(5)
>>> nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)
>>> nt
NestedTensor(size=(2, j1), offsets=tensor([0, 3, 8]), contiguous=True)
>>> nt
nt
>>> type(nt)
<class 'torch.nested._internal.nested_tensor.NestedTensor'>
>>> nt.share_memory_()
Segmentation fault (core dumped)