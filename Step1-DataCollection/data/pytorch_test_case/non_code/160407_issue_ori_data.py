def reset_parameters(self) -> None:
        if self.in1_features <= 0:
            raise ValueError(f"in1_features must be > 0, but got {self.in1_features}")
        bound = 1 / math.sqrt(self.weight.size(1))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)