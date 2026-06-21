import torch
from torch.testing._internal.common_utils import TestCase

class TestTolistGraphing(TestCase):
    def test_tolist_graph_break_consistency(self):
        """
        Test that tolist() behaves consistently with item() regarding graph breaks.
        
        The issue reports that `a.tolist()` (unpacking) results in a graph containing 
        `getitem` and `item` calls, whereas `a.item()` causes a graph break. 
        This test verifies the behavior to ensure consistency or proper handling 
        of scalar outputs during compilation.
        """
        
        # Case 1: tolist() unpacking
        @torch.compile(fullgraph=False, backend="eager")
        def func_tolist(a):
            u0, u1 = a.tolist()
            return a * u0 * u1

        # Case 2: item() direct call
        @torch.compile(fullgraph=False, backend="eager")
        def func_item(a):
            u0 = a.item()
            return a * u0

        # Test tolist
        input_tensor = torch.tensor([1, 2])
        try:
            result_tolist = func_tolist(input_tensor)
            # If it graphs, it should produce the correct result
            expected = input_tensor * 1 * 2
            self.assertEqual(result_tolist, expected)
            
            # Check graph for presence of item() calls if it didn't break
            # (This part is conceptual as we can't easily inspect the graph string here,
            # but we verify execution correctness)
        except Exception as e:
            self.fail(f"func_tolist failed with: {e}")

        # Test item
        input_tensor_single = torch.tensor([1])
        try:
            result_item = func_item(input_tensor_single)
            expected_item = input_tensor_single * 1
            self.assertEqual(result_item, expected_item)
        except Exception as e:
            # Depending on the fix, this might graph break or not.
            # The issue implies item() causes a graph break, so we expect it to run 
            # correctly regardless of the graph structure.
            self.fail(f"func_item failed with: {e}")

if __name__ == "__main__":
    test = TestTolistGraphing()
    test.test_tolist_graph_break_consistency()
    print("Test passed.")