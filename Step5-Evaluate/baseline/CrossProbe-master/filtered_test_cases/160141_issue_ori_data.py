import torch
model = torch.nn.LSTM(input_size=10, hidden_size=20, num_layers=1, dropout=0.2)
input_tensor = torch.randn(5, 3, 10)
output, _ = model(input_tensor)