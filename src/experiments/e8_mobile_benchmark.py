"""Experiment E8 Implementation: Mobile Edge Benchmark & Quantization Evaluation (Prompt 88)."""

import os
import json
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from src.experiments.base_experiment import BaseExperiment
from src.deployment.quantization_evaluator import QuantizationEvaluator, QuantizationReport
from src.models.landmark_baseline import LandmarkSequenceBaseline
from src.utils.config import BiISLConfig


class E8MobileBenchmarkExperiment(BaseExperiment):
    """Experiment E8: Mobile Edge Benchmark & Model Quantization Evaluation (Prompt 88)."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[BiISLConfig] = None
    ):
        super().__init__("E8", "Mobile Edge Benchmark Experiment", config_path, config)
        self.evaluator: Optional[QuantizationEvaluator] = None
        self.model: Optional[nn.Module] = None
        self.sample_inputs: Optional[torch.Tensor] = None
        self.target_outputs: Optional[torch.Tensor] = None

    def setup(self) -> None:
        """Set up model, test input data, and quantization evaluator."""
        self.logger.info("Setting up Experiment E8: Mobile Edge Benchmark Experiment...")
        
        # Instantiate full-precision PyTorch model
        self.model = LandmarkSequenceBaseline(
            input_dim=128,
            hidden_dim=64,
            num_layers=2,
            vocab_size=50
        )
        self.model.eval()

        # Create sample benchmark inputs (batch_size=16, seq_len=30, feature_dim=128)
        torch.manual_seed(self.config.training.seed)
        self.sample_inputs = torch.randn(16, 30, 128)
        with torch.no_grad():
            self.target_outputs = self.model(self.sample_inputs)

        # Initialize QuantizationEvaluator
        self.evaluator = QuantizationEvaluator(config=self.config)

    def run(self) -> Dict[str, Any]:
        """Execute quantization benchmark across FP32, FP16, and INT8 variants (Prompt 88)."""
        if self.model is None or self.evaluator is None or self.sample_inputs is None or self.target_outputs is None:
            self.setup()

        self.logger.info("Executing E8 Quantization Evaluation...")
        report: QuantizationReport = self.evaluator.evaluate_all_variants(
            fp32_model=self.model,
            sample_inputs=self.sample_inputs,
            target_outputs=self.target_outputs
        )

        results_dict = report.to_dict()

        # Log metrics to tracker
        for v in [report.baseline] + report.variants:
            prefix = v.variant_name.lower().replace(" ", "_")
            self.tracker.log_evaluation_results({
                f"{prefix}_quality": v.quality,
                f"{prefix}_quality_loss_pct": v.quality_loss_pct,
                f"{prefix}_p50_latency_ms": v.p50_latency_ms,
                f"{prefix}_p95_latency_ms": v.p95_latency_ms,
                f"{prefix}_ram_mb": v.ram_mb,
                f"{prefix}_size_mb": v.size_mb,
                f"{prefix}_energy_proxy": v.energy_proxy,
                f"{prefix}_accepted": 1.0 if v.accepted else 0.0
            })

        # Save JSON artifact report
        output_dir = getattr(self.config.experiment, "output_dir", "./artifacts/e8_mobile_benchmark")
        os.makedirs(output_dir, exist_ok=True)
        artifact_path = os.path.join(output_dir, "quantization_report.json")
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)

        self.logger.info(f"Quantization evaluation complete. Saved report to {artifact_path}")
        return results_dict

    def teardown(self) -> None:
        """Teardown experiment resources."""
        self.logger.info("Tearing down Experiment E8...")
        self.model = None
        self.sample_inputs = None
        self.target_outputs = None
        self.evaluator = None


if __name__ == "__main__":
    exp = E8MobileBenchmarkExperiment()
    exp.setup()
    results = exp.run()
    exp.teardown()
    print("Experiment E8 Execution Finished Successfully.")
