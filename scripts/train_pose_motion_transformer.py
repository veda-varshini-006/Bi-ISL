#!/usr/bin/env python3
"""Bi-ISL Pose-Motion Transformer Training CLI Script.

Usage:
    python scripts/train_pose_motion_transformer.py --epochs 5 --synthetic-fallback
    python scripts/train_pose_motion_transformer.py --data-dir data/processed/cislr --epochs 20
"""

import argparse
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.pose_motion_transformer import PoseMotionTranslationModel, BiISLTranslator
from src.training.pose_motion_trainer import PoseMotionTrainer, ContinuousISLDataset

try:
    import sentencepiece as spm
except ImportError:
    spm = None


def generate_synthetic_continuous_samples(num_samples: int = 32) -> list:
    """Generates synthetic continuous ISL sentence samples for pipeline verification."""
    sentences = [
        "What time is my doctor appointment today?",
        "I will go to the train station tomorrow.",
        "Please help me find the city hospital.",
        "My family lives in this city.",
        "Do you need food and water?",
        "Good morning thank you very much.",
        "Where is the nearest bus station?",
        "I am learning Indian Sign Language."
    ]

    samples = []
    for i in range(num_samples):
        # Generate 128 frames of 75 3D keypoints with sinusoidal motion
        t_steps = np.linspace(0, 4 * np.pi, 128)
        kps = np.zeros((128, 75, 3), dtype=np.float32)
        for f in range(128):
            kps[f, :, 0] = np.sin(t_steps[f] + np.linspace(0, 1, 75)) * 0.2
            kps[f, :, 1] = np.cos(t_steps[f] + np.linspace(0, 1, 75)) * 0.2
            kps[f, :, 2] = 0.5 + np.sin(t_steps[f] * 0.5) * 0.05

        text = sentences[i % len(sentences)]
        samples.append({
            "sample_id": f"sample_{i:04d}",
            "landmarks": kps,
            "text": text
        })

    return samples


def load_dataset_samples(data_dir: Path) -> list:
    """Loads continuous ISL dataset manifest from directory."""
    manifest_file = data_dir / "dataset_manifest.json"
    if not manifest_file.exists():
        manifest_file = data_dir / "manifest.json"

    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found in {data_dir}. Expected dataset_manifest.json.")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    samples = manifest_data.get("samples", manifest_data)
    loaded = []
    for s in samples:
        lm_path = data_dir / s.get("landmarks_path", s.get("landmarks", ""))
        text = s.get("english_sentence", s.get("text", ""))
        if lm_path.exists():
            loaded.append({"sample_id": s.get("sample_id"), "landmarks": str(lm_path), "text": text})

    return loaded


def main():
    parser = argparse.ArgumentParser(description="Train Bi-ISL Pose-Motion Transformer Model")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to continuous ISL dataset folder")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--tokenizer", type=str, default="artifacts/checkpoints/BiISL_SentencePiece_5000.model", help="SentencePiece model")
    parser.add_argument("--output-checkpoint", type=str, default="artifacts/checkpoints/PositionMotionTransformer_finetune_lr1e5_latest.pt", help="Checkpoint output path")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--synthetic-fallback", action="store_true", help="Use synthetic dataset if raw dataset is not downloaded")

    args = parser.parse_args()

    print("=" * 70)
    print("      Bi-ISL AR: Continuous ISL Translation Model Training")
    print("=" * 70)

    # 1. Load Tokenizer
    sp = None
    tok_path = Path(args.tokenizer)
    if tok_path.exists() and spm is not None:
        sp = spm.SentencePieceProcessor(model_file=str(tok_path))
        print(f"[Info] Loaded SentencePiece tokenizer from {tok_path} (Vocab size: {sp.get_piece_size()})")
    else:
        print("[Warning] SentencePiece tokenizer not found or not installed. Using fallback tokenization.")

    # 2. Prepare Data
    if args.data_dir and Path(args.data_dir).exists():
        print(f"[Info] Loading dataset from {args.data_dir}...")
        try:
            samples = load_dataset_samples(Path(args.data_dir))
            print(f"[Info] Found {len(samples)} valid continuous ISL samples.")
        except Exception as e:
            if args.synthetic_fallback:
                print(f"[Warning] Failed loading {args.data_dir} ({e}). Falling back to synthetic verification data.")
                samples = generate_synthetic_continuous_samples(32)
            else:
                raise e
    elif args.synthetic_fallback or not args.data_dir:
        print("[Info] No dataset directory specified. Utilizing continuous ISL training verification samples (--synthetic-fallback).")
        samples = generate_synthetic_continuous_samples(32)
    else:
        print(f"[Error] Directory not found: {args.data_dir}")
        sys.exit(1)

    # Split train/val
    split_idx = max(1, int(len(samples) * 0.8))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:] if split_idx < len(samples) else samples[:1]

    train_ds = ContinuousISLDataset(train_samples, max_seq_len=128, tokenizer=sp)
    val_ds = ContinuousISLDataset(val_samples, max_seq_len=128, tokenizer=sp)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Model & Trainer
    model = PoseMotionTranslationModel(vocab_size=5000, d_model=256)
    trainer = PoseMotionTrainer(
        model=model,
        learning_rate=args.lr,
        device=args.device
    )

    print(f"[Info] Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"[Info] Training on {len(train_samples)} samples for {args.epochs} epochs...")

    # 4. Training Loop
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(args.output_checkpoint, epoch=epoch, val_loss=val_loss)

    print("-" * 70)
    print(f"[Success] Training complete. Best checkpoint saved to {args.output_checkpoint}")


if __name__ == "__main__":
    main()
