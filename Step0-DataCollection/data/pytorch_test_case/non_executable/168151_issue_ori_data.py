#include <iostream>
#include <torch/torch.h>

int main()
{
    float ptr[1] = {0};
    torch::Tensor input = torch::from_blob(ptr, {1, 1, 1, 1}).toType(torch::kFloat32);
    input = input.to(torch::kCUDA);
    std::cout << input << std::endl;
    return 0;
}