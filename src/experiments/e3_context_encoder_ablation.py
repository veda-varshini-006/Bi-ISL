"""Experiment E3: Context Encoder Architecture Ablation Study (Prompt 33)."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import torch

from src.context.sbds_schema import SharedBidirectionalDialogueState, Entity, EntityType, Intent
from src.context.context_encoder import SBDSContextEncoder, EncoderArchitecture


def run_context_encoder_ablation_experiment(
    output_dir: str = "./artifacts/reports/context"
) -> Tuple[str, str, Dict[str, Any]]:
    """Run ablation comparison between SIMPLE_EMBEDDING and TRANSFORMER_ATTENTION context encoders."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dummy_states = [
        SharedBidirectionalDialogueState(
            sequence_id="seq_e3_1",
            active_entities=[Entity(entity_id="e1", name="Doctor", entity_type=EntityType.PERSON)],
            dialogue_intent=Intent(intent_type="HEALTH_CHECK", domain="MEDICAL")
        ),
        SharedBidirectionalDialogueState(
            sequence_id="seq_e3_2",
            active_entities=[]
        )
    ]

    architectures = [
        EncoderArchitecture.SIMPLE_EMBEDDING,
        EncoderArchitecture.TRANSFORMER_ATTENTION
    ]

    ablation_results = {}

    for arch in architectures:
        model = SBDSContextEncoder(embed_dim=256, architecture=arch)
        num_params = sum(p.numel() for p in model.parameters())

        start_time = time.perf_counter()
        for _ in range(50):
            out_tensor, mask_tensor = model(dummy_states)
        latency_ms = round(((time.perf_counter() - start_time) / 50.0) * 1000.0, 3)

        ablation_results[arch.value] = {
            "parameter_count": num_params,
            "inference_latency_ms": latency_ms,
            "output_tensor_shape": list(out_tensor.shape),
            "mask_tensor_shape": list(mask_tensor.shape)
        }

    json_path = out_path / "context_encoder_ablation.json"
    md_path = out_path / "context_encoder_ablation.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    md_lines = [
        "# Bi-ISL Context Encoder Architecture Ablation Study (Prompt 33)",
        "",
        "| Architecture | Parameter Count | Latency (ms) | Output Shape | Mask Shape |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]

    for name, r in ablation_results.items():
        md_lines.append(
            f"| **{name}** | {r['parameter_count']:,} | {r['inference_latency_ms']} ms | `{r['output_tensor_shape']}` | `{r['mask_tensor_shape']}` |"
        )

    md_lines.extend([
        "",
        "✅ **Compact context encoder ablation completed.** No large generative LLM used."
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return str(json_path), str(md_path), ablation_results


if __name__ == "__main__":
    run_context_encoder_ablation_experiment()
