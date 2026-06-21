import unittest
import torch
from torch.optim import AdamW
from omegaconf import OmegaConf

class TestOptimizerBetasValidation(unittest.TestCase):
    def test_betas_rejects_omegaconf_listconfig(self):
        """
        Test that AdamW validates the 'betas' parameter type.
        It should raise a TypeError when an OmegaConf ListConfig is passed
        to prevent unexpected serialization behavior.
        """
        # Create a simple model
        model = torch.nn.Linear(10, 1)
        
        # Create a config with nested structure containing betas
        cfg = OmegaConf.create({
            'model': {'betas': [0.9, 0.999]},
            'data': {'batch_size': 32}
        })

        # Expect a TypeError when passing OmegaConf ListConfig
        with self.assertRaises(TypeError):
            AdamW(model.parameters(), lr=1e-3, betas=cfg.model.betas)

    def test_betas_accepts_list_and_tuple(self):
        """
        Test that AdamW accepts standard list and tuple types for 'betas'.
        """
        model = torch.nn.Linear(10, 1)
        
        # Should work with list
        opt1 = AdamW(model.parameters(), lr=1e-3, betas=[0.9, 0.999])
        self.assertIsInstance(opt1, AdamW)
        
        # Should work with tuple
        opt2 = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.assertIsInstance(opt2, AdamW)

if __name__ == '__main__':
    unittest.main()