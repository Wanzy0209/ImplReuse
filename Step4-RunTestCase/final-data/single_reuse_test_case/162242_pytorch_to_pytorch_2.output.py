import numpy as np
import torch


def test_scatter_add_deterministic():
    # Recalculated ground truth for torch.scatter_add
    # Logic: result[b, index[b, i, k], k] += src[b, i, k]
    # Batch 0:
    # inputs[0] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    # src[0] = [[-99, -98, -97], [-96, -95, -94]]
    # index[0] = [[0, 1, 0], [1, 1, 0]]
    # 
    # k=0: inputs[0, 0, 0] += -99; inputs[0, 1, 0] += -96 -> [-99, -92, 0, 0] (cols)
    # k=1: inputs[0, 1, 1] += -98; inputs[0, 1, 1] += -95 -> [0, -188, 0, 0] (cols)
    # k=2: inputs[0, 0, 2] += -97; inputs[0, 0, 2] += -94 -> [-189, 0, 0, 0] (cols)
    # k=3: No updates -> [0, 0, 0, 0]
    #
    # Result Batch 0:
    # [[-99, 1, -189, 3],
    #  [-92, -188, 6, 7],
    #  [8, 9, 10, 11]]
    #
    # Batch 1:
    # inputs[1] = [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
    # src[1] = [[-93, -92, -91], [-90, -89, -88]]
    # index[1] = [[1, 0, 1], [0, 0, 1]]
    #
    # k=0: inputs[1, 1, 0] += -93; inputs[1, 0, 0] += -90 -> [-78, -77, 0, 0]
    # k=1: inputs[1, 0, 1] += -92; inputs[1, 0, 1] += -89 -> [-168, 0, 0, 0]
    # k=2: inputs[1, 1, 2] += -91; inputs[1, 1, 2] += -88 -> [0, -161, 0, 0]
    # k=3: No updates
    #
    # Result Batch 1:
    # [[-78, -168, 14, 15],
    #  [-77, 17, -161, 19],
    #  [20, 21, 22, 23]]

    gt_res = np.array(
        [
            [
                [-99.0, 1.0, -189.0, 3.0],
                [-92.0, -188.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            [
                [-78.0, -168.0, 14.0, 15.0],
                [-77.0, 17.0, -161.0, 19.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
        ],
        dtype=np.float32,
    )
    
    # For scatter_add, all elements in input contribute to the output (as the base value).
    # Therefore, the gradient w.r.t. input is the gradient of the output.
    gt_input_grad = np.ones((2, 3, 4), dtype=np.float32)
    
    # For scatter_add, all elements in src are added to the output exactly once.
    # Therefore, the gradient w.r.t. src is the gradient of the output scattered to the src indices.
    # Since output grad is ones, src grad is ones.
    gt_src_grad = np.ones((2, 2, 3), dtype=np.float32)

    for i in range(1000):
        torch.cuda.empty_cache()
        inputs = torch.arange(24, dtype=torch.float32).reshape([2, 3, 4]).cuda()
        src = torch.arange(-99, -99 + (2 * 2 * 3), dtype=torch.float32).reshape(
            [2, 2, 3]
        ).cuda()
        index = torch.tensor(
            [
                [
                    [0, 1, 0],
                    [1, 1, 0],
                ],
                [
                    [1, 0, 1],
                    [0, 0, 1],
                ],
            ],
            dtype=torch.int64,
        ).cuda()

        inputs.requires_grad = True
        src.requires_grad = True

        # Changed from torch.scatter to torch.scatter_add
        res = torch.scatter_add(inputs, 1, index, src)
        res.backward(torch.ones_like(res))

        print(f"Test {i + 1}/{1000}")
        np.testing.assert_allclose(res.cpu().detach().numpy(), gt_res)
        np.testing.assert_allclose(inputs.grad.cpu().numpy(), gt_input_grad)
        np.testing.assert_allclose(src.grad.cpu().numpy(), gt_src_grad)


if __name__ == "__main__":
    test_scatter_add_deterministic()