"""Experiment E0: Dataset Leakage Audit Runner for Bi-ISL."""

from typing import Dict, List, Any, Optional
from src.data.schema import CanonicalDataSample
from src.data.split_auditor import SplitAuditor, LeakageAuditFailedError
from src.experiments.base_experiment import BaseExperiment


class E0LeakageAuditExperiment(BaseExperiment):
    """Experiment E0 runner executing dataset leakage audit across train/dev/test splits."""

    def __init__(self, config=None, experiment_id: str = "E0"):
        super().__init__(exp_id=experiment_id, title="E0 Dataset Leakage Audit", config=config)
        self.auditor = SplitAuditor(logger=self.logger)

    def setup(self) -> None:
        """Setup experiment resources."""
        self.logger.info(f"Setting up Experiment {self.experiment_id}...")

    def teardown(self) -> None:
        """Clean up experiment resources."""
        self.logger.info(f"Teardown Experiment {self.experiment_id}.")

    def run(self, splits: Optional[Dict[str, List[CanonicalDataSample]]] = None) -> Dict[str, Any]:
        """Execute Experiment E0 dataset leakage audit."""
        self.logger.info("Executing Experiment E0: Dataset Leakage Audit...")

        if splits is None:
            # Generate clean dummy benchmark splits for audit baseline
            splits = {
                "train": [
                    CanonicalDataSample(sample_id="tr_01", dataset="INCLUDE", source_video_id="v_101", signer_id="s_01", text="hello"),
                    CanonicalDataSample(sample_id="tr_02", dataset="INCLUDE", source_video_id="v_102", signer_id="s_02", text="world")
                ],
                "dev": [
                    CanonicalDataSample(sample_id="dev_01", dataset="INCLUDE", source_video_id="v_201", signer_id="s_03", text="good morning")
                ],
                "test": [
                    CanonicalDataSample(sample_id="te_01", dataset="INCLUDE", source_video_id="v_301", signer_id="s_04", text="thank you")
                ]
            }

        try:
            summary = self.auditor.audit_splits(splits=splits, signer_disjoint_required=True)
            jfile, cfile, mfile = self.auditor.export_reports(summary, output_dir=f"./artifacts/runs/{self.experiment_id}/reports")

            self.tracker.log_evaluation_results({
                "total_samples": summary["total_samples_audited"],
                "total_violations": summary["total_violations"],
                "critical_violations": summary["critical_violations"]
            })
            self.tracker.finish(status="completed")

            return summary

        except LeakageAuditFailedError as e:
            self.tracker.log_error(str(e))
            self.tracker.finish(status="failed")
            raise
