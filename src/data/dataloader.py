"""Bi-ISL Custom Collation & DataLoader Construction (Prompt 21)."""

from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class BiISLCollate:
    """Custom collation function padding variable-length sequences and target token sequences in a batch."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids = [item["sample_id"] for item in batch]
        target_texts = [item["target_text"] for item in batch]
        valid_lengths = [item["valid_length"] for item in batch]

        attention_masks = torch.stack([item["attention_mask"] for item in batch], dim=0)

        rgb_batch = None
        if batch[0]["rgb"] is not None:
            rgb_batch = torch.stack([item["rgb"] for item in batch], dim=0)

        landmark_batch = None
        if batch[0]["landmark"] is not None:
            landmark_batch = torch.stack([item["landmark"] for item in batch], dim=0)

        target_list = [item["target_tokens"] for item in batch]
        max_target_len = max(len(t) for t in target_list)
        padded_targets = torch.full((len(batch), max_target_len), self.pad_token_id, dtype=torch.long)
        target_masks = torch.zeros((len(batch), max_target_len), dtype=torch.bool)

        for i, t in enumerate(target_list):
            padded_targets[i, :len(t)] = t
            target_masks[i, :len(t)] = True

        return {
            "sample_ids": sample_ids,
            "rgb": rgb_batch,
            "landmark": landmark_batch,
            "attention_mask": attention_masks,
            "target_tokens": padded_targets,
            "target_mask": target_masks,
            "target_texts": target_texts,
            "valid_lengths": valid_lengths
        }


def create_biisl_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    is_distributed: bool = False,
    seed: int = 42
) -> DataLoader:
    """Create PyTorch DataLoader with custom collation and DDP sampler support."""
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
        shuffle = False

    collate_fn = BiISLCollate()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
