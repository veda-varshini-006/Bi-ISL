"""Experiment E1 Runner comparing GRU vs BiLSTM Landmark Sequence Baselines (Prompt 22)."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple

from src.data.dataset import SyntheticISLDataset
from src.data.dataloader import create_biisl_dataloader
from src.models.landmark_baseline import LandmarkSequenceBaseline, LandmarkBaselineTrainer


def run_e1_landmark_experiment(
    epochs: int = 2,
    batch_size: int = 4,
    output_dir: str = "./artifacts/reports/baselines"
) -> Tuple[str, str]:
    """Train and compare GRU vs BiLSTM landmark sequence baselines, exporting Markdown report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Datasets & Loaders
    train_dataset = SyntheticISLDataset(num_samples=16, modality="landmark", max_seq_len=32)
    val_dataset = SyntheticISLDataset(num_samples=8, modality="landmark", max_seq_len=32)

    train_loader = create_biisl_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_biisl_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    results: Dict[str, Any] = {}

    for rnn_type in ["GRU", "BiLSTM"]:
        model = LandmarkSequenceBaseline(input_dim=258, hidden_dim=64, num_layers=2, vocab_size=50, rnn_type=rnn_type)
        trainer = LandmarkBaselineTrainer(model, lr=1e-3, checkpoint_dir=f"./artifacts/checkpoints/e1_{rnn_type.lower()}")

        params = model.count_parameters()
        latency_ms = model.measure_inference_latency(batch_size=1, seq_len=32, num_runs=10)

        val_loss = 0.0
        for ep in range(1, epochs + 1):
            _ = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)
            trainer.save_checkpoint(epoch=ep, val_loss=val_loss)

        results[rnn_type] = {
            "rnn_type": rnn_type,
            "parameters": params,
            "latency_ms": latency_ms,
            "final_val_loss": val_loss
        }

    json_rep = out_path / "landmarks_baseline_comparison.json"
    md_rep = out_path / "landmarks_baseline_comparison.md"

    with open(json_rep, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_lines = [
        "# Experiment E1: Landmark Sequence Baseline Comparison (GRU vs BiLSTM)",
        "",
        "| Architecture | Trainable Parameters | Inference Latency (ms / sample) | Validation Loss |",
        "| :--- | :---: | :---: | :---: |"
    ]
    for name, r in results.items():
        md_lines.append(f"| **{name}** | {r['parameters']:,} | {r['latency_ms']} ms | {r['final_val_loss']} |")

    md_lines.extend([
        "",
        "✅ **Diagnostic sequence baselines operational with sequence masking, parameter logging, and atomic checkpointing.**"
    ])

    with open(md_rep, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return str(json_rep), str(md_rep)


if __name__ == "__main__":
    run_e1_landmark_experiment()
