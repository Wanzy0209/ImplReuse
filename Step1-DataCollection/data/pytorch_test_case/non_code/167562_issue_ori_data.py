x = torch.ones(1, 2, 3, 4, 5)
    print(x.sum(dim=torch.tensor(3)))  # Ed, why are we accepting a scalar tensor here?!
    print(x.sum(dim=3))