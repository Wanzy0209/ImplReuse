x = torch.zeros(74, 32, 30090, 81, device=torch.device("cuda"), dtype=torch.bfloat16)
torch.nn.functional.max_pool2d(x, kernel_size=(1,2), stride=(1,2), ceil_mode=False, padding=0)