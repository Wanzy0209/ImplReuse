def test_error_on_graph_break_nonempty_checkpoint(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            x = x + 1
            x = x + 1
            x = x + 1
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 1

        with self.assertRaises(Unsupported):
            fn(torch.ones(3))

        self.assertEqual(cnts.frame_count, 0)