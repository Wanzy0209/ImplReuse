flag = True
        dummy = lambda: None

        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            x = x + 2
            if flag:
                dummy.attr0 = x
            else:
                with torch.no_grad():
                    dummy.attr1 = x
            return x + 4

        inp = torch.ones(3)
        opt_fn = torch.compile(fn, backend="eager")
        assert torch.allclose(fn(inp), opt_fn(inp))
        flag = False
        assert torch.allclose(fn(inp), opt_fn(inp))