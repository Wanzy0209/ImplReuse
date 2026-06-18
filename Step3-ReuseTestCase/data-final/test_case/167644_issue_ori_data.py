import torch

def cli_main():
    torch.set_float32_matmul_precision("high")

if __name__ == '__main__':
    cli_main()