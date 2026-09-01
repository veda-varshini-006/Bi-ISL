"""Smoke tests for checkpoint save and load."""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

def test_checkpoint_save_load_smoke():
    model_orig = DummyNet()
    model_loaded = DummyNet()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "checkpoint.pt")
        
        # Save checkpoint
        torch.save({
            "epoch": 5,
            "model_state_dict": model_orig.state_dict(),
            "optimizer_state_dict": {}
        }, ckpt_path)
        
        assert os.path.exists(ckpt_path)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, weights_only=True)
        model_loaded.load_state_dict(ckpt["model_state_dict"])

        assert ckpt["epoch"] == 5

        # Verify weights match
        for p1, p2 in zip(model_orig.parameters(), model_loaded.parameters()):
            assert torch.allclose(p1, p2)
