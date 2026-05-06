>>> # torch.min -> reduce over all dimensions
>>> a = torch.ones([5]).cuda()
>>> a.requires_grad=True
>>> min_val = torch.min(a)
>>> min_val.backward()
>>> a.grad
tensor([0.2000, 0.2000, 0.2000, 0.2000, 0.2000], device='cuda:0')
...
>>> # torch.min(input, dim=...) -> reduce over specified dimension
>>> a = torch.ones([5]).cuda()
>>> a.requires_grad=True
>>> min_val = torch.min(a, dim=0)
>>> min_val.values.backward()
>>> a.grad
tensor([1., 0., 0., 0., 0.], device='cuda:0')