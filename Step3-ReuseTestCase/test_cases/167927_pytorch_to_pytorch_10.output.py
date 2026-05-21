import torch
import torch.distributed as dist
import os

def test_gather_object_with_fullgraph():
    """
    Test case to verify if torch.distributed.gather_object works correctly
    when wrapped in torch.compiler.disable inside a torch.compile(fullgraph=True) block.
    This addresses the issue where fullgraph=True raises an error on disabled functions.
    """
    if not dist.is_available():
        print("Skipping test: torch.distributed not available")
        return

    # Setup minimal distributed environment (single process)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    try:
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

        def disabled_gather(obj):
            # Intentionally disable compilation for this function
            with torch.compiler.disable():
                output_list = [None] * dist.get_world_size()
                dist.gather_object(obj, output_list, dst=0)
                return output_list

        def main_func():
            return disabled_gather("test_data")

        # Compile with fullgraph=True
        # Bug: This used to raise torch._dynamo.exc.Unsupported
        # Expected: It should run successfully, respecting the disable context
        compiled_main = torch.compile(main_func, fullgraph=True)
        result = compiled_main()

        print(f"Test passed. Result: {result}")
        assert result == ["test_data"], f"Expected ['test_data'], got {result}"

    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    test_gather_object_with_fullgraph()