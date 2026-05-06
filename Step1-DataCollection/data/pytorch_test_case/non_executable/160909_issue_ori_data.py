def my_backend():
    // my compile logic

model = Model()
compiled = torch.compile(model.to("PrivateUse1"), backend=my_backend)
result = compiled(data.to('PrivateUse1'))