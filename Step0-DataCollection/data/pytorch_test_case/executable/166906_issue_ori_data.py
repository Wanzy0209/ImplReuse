def test_dynamic_annotate(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                with torch.fx.traceback.annotate({"moo": 0}):
                    x = torch.cat([x, x])
                    b = y.item()
                    torch._check(b >= x.shape[0])
                    return x * b
        
        with torch.fx.traceback.preserve_node_meta():
            ep = torch.export.export(M(), (torch.randn(3), torch.tensor(6)), dynamic_shapes={"x": {0: Dim("b")}, "y": None})
        print(ep)
        for node in ep.graph.nodes:
            print(node, node.meta.get("custom"))