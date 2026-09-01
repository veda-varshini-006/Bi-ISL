"""CLI Script for Bi-ISL Video Dataset Validation and Quarantine (Prompt 17).

Usage:
    python scripts/data/validate_videos.py --data-dir ./data/raw/INCLUDE
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import CanonicalDataSample
from src.data.video_validator import VideoValidator


def main():
    parser = argparse.ArgumentParser(description="Bi-ISL Video Dataset Integrity & Quarantine CLI")
    parser.add_argument("--data-dir", type=str, default="./data/raw", help="Path to raw video directory")
    parser.add_argument("--quarantine-dir", type=str, default="./data/quarantine", help="Quarantine output directory")
    parser.add_argument("--report-dir", type=str, default="./artifacts/reports/data_health", help="Report output directory")
    parser.add_argument("--no-quarantine", action="store_true", help="Disable automatic quarantine moving")
    args = parser.parse_args()

    validator = VideoValidator(quarantine_dir=args.quarantine_dir)
    data_path = Path(args.data_dir)

    video_files = list(data_path.rglob("*.mp4")) + list(data_path.rglob("*.avi"))
    print(f"Discovered {len(video_files)} video files under '{args.data_dir}'...")

    samples = [
        CanonicalDataSample(sample_id=f"v_{i}", dataset="AUDIT", video_path=str(vpath))
        for i, vpath in enumerate(video_files)
    ]

    stats, results = validator.validate_dataset_videos(samples, quarantine_corrupt=not args.no_quarantine)
    json_rep, md_rep = validator.export_health_reports(stats, results, output_dir=args.report_dir)

    print("=" * 60)
    print("BI-ISL VIDEO HEALTH REPORT SUMMARY")
    print("=" * 60)
    print(f"Total Videos Audited:   {stats['total_videos_audited']}")
    print(f"Valid Videos:           {stats['valid_videos_count']}")
    print(f"Quarantined Videos:     {stats['quarantined_videos_count']}")
    print(f"Total Duration (Hours): {stats['total_duration_hours']}")
    print(f"Health Percentage:      {stats['health_percentage']}%")
    print(f"Markdown Report:        {md_rep}")
    print("=" * 60)


if __name__ == "__main__":
    main()
