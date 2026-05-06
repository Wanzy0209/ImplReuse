def test_init_with_user_generator(self):
        device_mesh = self.build_device_mesh()
        torch.manual_seed(42)
        rng = torch.Generator(device="cuda").manual_seed(42)
        t1 = torch.distributed.tensor.empty(
            (2, 3), device_mesh=device_mesh, placements=[Shard(0)]
        )
        t2 = torch.distributed.tensor.empty(
            (2, 3), device_mesh=device_mesh, placements=[Shard(0)]
        )
        for i in range(2):
            print(f"{i=}")
            # run a second time, to make sure that `rng`'s offset-state is advancing on the second usage
            torch.nn.init.uniform_(t1, 0.0, 1.0)
            torch.nn.init.uniform_(t2, 0.0, 1.0, rng)
            self.assertEqual(t1.full_tensor(), t2.full_tensor(), f"Failed at {i=}")

        # Problem area:
        # (1)
        # torch.distributed.tensor._random._rng_tracker._manual_seed(55)
        # torch.manual_seed(55)
        # (2)
        # rng.manual_seed(55)
        torch.nn.init.uniform_(t1, 0.0, 1.0)
        torch.nn.init.uniform_(t2, 0.0, 1.0, rng)
        self.assertEqual(t1.full_tensor(), t2.full_tensor())