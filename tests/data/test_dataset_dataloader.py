"""Unit tests for Bi-ISL PyTorch Dataset and DataLoader Subsystem."""

import pytest
import torch

from src.data.dataset import BiISLDataset, SyntheticISLDataset
from src.data.dataloader import BiISLCollate, create_biisl_dataloader


def test_synthetic_dataset_fixture():
    """Test SyntheticISLDataset creation and length."""
    dataset = SyntheticISLDataset(num_samples=10, modality="multimodal", max_seq_len=32)
    assert len(dataset) == 10

    item = dataset[0]
    assert item["sample_id"] == "synth_0"
    assert item["rgb"].shape == (32, 3, 224, 224)
    assert item["landmark"].shape == (32, 258)
    assert item["attention_mask"].shape == (32,)
    assert isinstance(item["target_tokens"], torch.Tensor)


def test_dataset_modalities():
    """Test RGB-only, Landmark-only, and Multimodal dataset modes."""
    ds_rgb = SyntheticISLDataset(num_samples=2, modality="rgb", max_seq_len=16)
    item_rgb = ds_rgb[0]
    assert item_rgb["rgb"] is not None
    assert item_rgb["landmark"] is None

    ds_lm = SyntheticISLDataset(num_samples=2, modality="landmark", max_seq_len=16)
    item_lm = ds_lm[0]
    assert item_lm["rgb"] is None
    assert item_lm["landmark"] is not None


def test_dataloader_batch_collation():
    """Test custom collation with variable-length batching and attention masks."""
    dataset = SyntheticISLDataset(num_samples=4, modality="multimodal", max_seq_len=24)
    loader = create_biisl_dataloader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    assert len(batch["sample_ids"]) == 2
    assert batch["rgb"].shape == (2, 24, 3, 224, 224)
    assert batch["landmark"].shape == (2, 24, 258)
    assert batch["attention_mask"].shape == (2, 24)
    assert batch["target_tokens"].ndim == 2
    assert batch["target_mask"].shape == batch["target_tokens"].shape


def test_create_biisl_dataloader_execution():
    """Test iterating over entire DataLoader dataset."""
    dataset = SyntheticISLDataset(num_samples=5, modality="landmark", max_seq_len=16)
    loader = create_biisl_dataloader(dataset, batch_size=2, shuffle=True)

    batch_count = 0
    total_samples = 0
    for batch in loader:
        batch_count += 1
        total_samples += len(batch["sample_ids"])

    assert batch_count == 3  # ceil(5 / 2) = 3
    assert total_samples == 5
