def test_namedtuple(self):
        from collections import namedtuple
        Point = namedtuple('Point', 'x y')
        
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y 
        
        inp = Point(torch.ones(3), torch.ones(3))
        print(M()(*inp))
        
        # errors
        ep = torch.export.export(M(), inp, strict=False)
        print(ep)

        # succeeds
        ep = torch.export.export(M(), inp, strict=True)
        print(ep)

        # workaround could be to convert namedtuple to a kwarg
        inp_kwargs =  {field: getattr(inp, field) for field in inp._fields}
        ep = torch.export.export(M(), (), inp_kwargs)
        print(ep)