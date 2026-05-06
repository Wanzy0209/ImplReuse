import torch

class Eps:
    def __init__(self, bitwidth):
        self.mantissa_bitwidth = 0
        self.exponent_bias = 0
        self.ulp = 0 # u
        self.half_ulp = 0 # u / 2
        self.smallest_larger_than_half_ulp = 0 # smallest representable number larger than 1
        self.dtype = None
        self.dtype_converted = None
        self.get_fp_params(bitwidth)
        self.get_smallest_larger_than_half_ulp()

    def get_fp_params(self, bitwidth):
        if bitwidth == 32:
            self.mantissa_bitwidth = 23
            self.exponent_bias = 127
            self.dtype = torch.float32
            self.dtype_converted = torch.int32
        elif bitwidth == 16:
            self.mantissa_bitwidth = 10
            self.exponent_bias = 15
            self.dtype = torch.float16
            self.dtype_converted = torch.int16
        else:
            raise ValueError(f"{bitwidth} is invalid!")

    def get_smallest_larger_than_half_ulp(self):
        self.ulp = torch.tensor([2 ** (-self.mantissa_bitwidth)], dtype=self.dtype)
        self.half_ulp = self.ulp / 2
        self.smallest_larger_than_half_ulp = self.half_ulp.view(self.dtype_converted) + 1
        self.smallest_larger_than_half_ulp = self.smallest_larger_than_half_ulp.view(self.dtype)

    def print_result(self):
        torch_eps = torch.finfo(self.dtype).eps
        print(f"dtype: {self.dtype}")
        print(f"torch.finfo({self.dtype}).eps is {torch_eps}, eps + 1.0 == 1.0? ==> {torch_eps + 1.0 == 1.0}")
        print(f"smallest number larger than u / 2 is {self.smallest_larger_than_half_ulp.item()}, my eps + 1.0 == 1.0? ==> {(self.smallest_larger_than_half_ulp + 1.0 == 1.0).item()}")
        print(f"u / 2 is {self.half_ulp.item()}, u / 2 + 1.0 == 1.0? ==> {(self.half_ulp + 1.0 == 1.0).item()}")

def main():
    fp32_result = Eps(32)
    fp16_reresult = Eps(16)
    fp32_result.print_result()
    fp16_reresult.print_result()

if __name__ == '__main__':
    main()