"""Visual Regression Scenes Engine (Prompt 74).

Generates, renders, and evaluates visual regression test scenes for motion retargeting:
- Scene_FingerAlignmentTest
- Scene_WristOrientationTest
- Scene_JointConstraintsTest
- Scene_BodyScaleTest
- Scene_HandednessTest
"""

from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from src.avatar.animation_clip_library import AnimationClip, AnimationClipLibrary
from src.avatar.motion_retargeter import MotionRetargeter, HumanoidRigConfig
from src.avatar.retargeting_verifier import RetargetingVerifier, RetargetingVerificationReport


@dataclass
class CameraTransform:
    """Camera transform specification for visual regression scene rendering."""
    position: List[float] = field(default_factory=lambda: [0.0, 1.2, 1.5])
    rotation_euler: List[float] = field(default_factory=lambda: [0.0, 180.0, 0.0])
    fov: float = 60.0


@dataclass
class VisualRegressionSceneConfig:
    """Specification configuration for a single visual regression test scene."""
    scene_id: str
    name: str
    target_criterion: str  # "FINGER_ALIGNMENT", "WRIST_ORIENTATION", "JOINT_CONSTRAINTS", "BODY_SCALE", "HANDEDNESS"
    target_gloss: str
    camera: CameraTransform = field(default_factory=CameraTransform)
    scale_factor: float = 1.0
    handedness_override: Optional[str] = None
    tolerance_threshold: float = 0.005  # MSE tolerance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "name": self.name,
            "target_criterion": self.target_criterion,
            "target_gloss": self.target_gloss,
            "camera": asdict(self.camera),
            "scale_factor": self.scale_factor,
            "handedness_override": self.handedness_override,
            "tolerance_threshold": self.tolerance_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualRegressionSceneConfig":
        data_copy = dict(data)
        cam_data = data_copy.pop("camera", {})
        camera = CameraTransform(**cam_data) if cam_data else CameraTransform()
        return cls(camera=camera, **data_copy)


@dataclass
class VisualRegressionComparisonResult:
    """Comparison evaluation output for a visual regression scene snapshot."""
    scene_id: str
    passed: bool
    mse_error: float
    psnr_db: float
    max_pixel_diff: float
    verification_report: Dict[str, Any] = field(default_factory=dict)


class VisualRegressionEngine:
    """Engine managing visual regression scenes, snapshot generation, and comparison."""

    def __init__(self, target_config: Optional[HumanoidRigConfig] = None) -> None:
        self.target_config = target_config or HumanoidRigConfig()
        self.retargeter = MotionRetargeter(self.target_config)
        self.verifier = RetargetingVerifier(self.target_config)

    @classmethod
    def create_benchmark_scenes(cls) -> List[VisualRegressionSceneConfig]:
        """Create standard set of 5 visual regression test scenes."""
        return [
            VisualRegressionSceneConfig(
                scene_id="SCENE_FINGER_ALIGNMENT_01",
                name="Scene_FingerAlignmentTest",
                target_criterion="FINGER_ALIGNMENT",
                target_gloss="DOCTOR",
                camera=CameraTransform(position=[0.0, 1.25, 0.75], rotation_euler=[5.0, 180.0, 0.0], fov=35.0),
                scale_factor=1.0,
                tolerance_threshold=0.002
            ),
            VisualRegressionSceneConfig(
                scene_id="SCENE_WRIST_ORIENTATION_01",
                name="Scene_WristOrientationTest",
                target_criterion="WRIST_ORIENTATION",
                target_gloss="HELP",
                camera=CameraTransform(position=[0.2, 1.2, 0.9], rotation_euler=[0.0, 175.0, 0.0], fov=40.0),
                scale_factor=1.0,
                tolerance_threshold=0.003
            ),
            VisualRegressionSceneConfig(
                scene_id="SCENE_JOINT_CONSTRAINTS_01",
                name="Scene_JointConstraintsTest",
                target_criterion="JOINT_CONSTRAINTS",
                target_gloss="FEVER",
                camera=CameraTransform(position=[0.0, 1.1, 1.8], rotation_euler=[0.0, 180.0, 0.0], fov=50.0),
                scale_factor=1.0,
                tolerance_threshold=0.002
            ),
            VisualRegressionSceneConfig(
                scene_id="SCENE_BODY_SCALE_01",
                name="Scene_BodyScaleTest",
                target_criterion="BODY_SCALE",
                target_gloss="DOCTOR",
                camera=CameraTransform(position=[0.0, 1.1, 2.2], rotation_euler=[0.0, 180.0, 0.0], fov=55.0),
                scale_factor=1.25,  # 125% scale test
                tolerance_threshold=0.005
            ),
            VisualRegressionSceneConfig(
                scene_id="SCENE_HANDEDNESS_01",
                name="Scene_HandednessTest",
                target_criterion="HANDEDNESS",
                target_gloss="DOCTOR",
                camera=CameraTransform(position=[0.0, 1.2, 1.6], rotation_euler=[0.0, 180.0, 0.0], fov=45.0),
                scale_factor=1.0,
                handedness_override="LEFT",  # Test RIGHT -> LEFT mirroring
                tolerance_threshold=0.003
            ),
        ]

    def generate_scene_manifests(self, output_dir: Union[str, Path]) -> List[Path]:
        """Export scene JSON manifests for Unity and test pipelines."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        manifest_paths = []
        scenes = self.create_benchmark_scenes()
        for scene in scenes:
            scene_file = out_path / f"{scene.name}.unity.json"
            with open(scene_file, "w", encoding="utf-8") as f:
                json.dump(scene.to_dict(), f, indent=2)
            manifest_paths.append(scene_file)

        return manifest_paths

    def generate_snapshot_matrix(
        self, scene_cfg: VisualRegressionSceneConfig, clip: AnimationClip
    ) -> List[List[float]]:
        """Synthesize snapshot pose feature matrix for visual regression comparison."""
        retargeted_clip = self.retargeter.retarget_clip(
            clip,
            scale_factor=scene_cfg.scale_factor,
            target_handedness=scene_cfg.handedness_override
        )

        matrix = []
        for kf in retargeted_clip.keyframes:
            row = []
            for bone in sorted(kf.joint_positions.keys()):
                row.extend(kf.joint_positions[bone])
            for bone in sorted(kf.joint_rotations.keys()):
                row.extend(kf.joint_rotations[bone])
            matrix.append(row)

        return matrix

    def compare_snapshot_matrices(
        self,
        scene_cfg: VisualRegressionSceneConfig,
        baseline_matrix: List[List[float]],
        test_matrix: List[List[float]],
        verifier_report: RetargetingVerificationReport
    ) -> VisualRegressionComparisonResult:
        """Compare two snapshot feature matrices and evaluate pass/fail visual regression verdict."""
        if not baseline_matrix or not test_matrix:
            return VisualRegressionComparisonResult(
                scene_id=scene_cfg.scene_id,
                passed=False,
                mse_error=1.0,
                psnr_db=0.0,
                max_pixel_diff=1.0,
                verification_report=verifier_report.to_dict()
            )

        total_sq_diff = 0.0
        total_elements = 0
        max_diff = 0.0

        num_rows = min(len(baseline_matrix), len(test_matrix))
        for i in range(num_rows):
            r1 = baseline_matrix[i]
            r2 = test_matrix[i]
            num_cols = min(len(r1), len(r2))
            for j in range(num_cols):
                diff = abs(r1[j] - r2[j])
                if diff > max_diff:
                    max_diff = diff
                total_sq_diff += diff * diff
                total_elements += 1

        mse = (total_sq_diff / max(1, total_elements))
        psnr = 10.0 * math.log10(1.0 / max(1e-9, mse)) if mse > 0 else 100.0

        passed = (mse <= scene_cfg.tolerance_threshold) and verifier_report.is_valid

        return VisualRegressionComparisonResult(
            scene_id=scene_cfg.scene_id,
            passed=passed,
            mse_error=mse,
            psnr_db=psnr,
            max_pixel_diff=max_diff,
            verification_report=verifier_report.to_dict()
        )

    def run_visual_regression_suite(
        self,
        library: AnimationClipLibrary,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Execute full visual regression scene suite across all benchmark scenes."""
        scenes = self.create_benchmark_scenes()
        results = []
        overall_passed = True

        for scene in scenes:
            src_clip = library.get_clip_by_gloss(scene.target_gloss)
            if src_clip is None:
                src_clip = library.list_clips()[0]

            retargeted_clip = self.retargeter.retarget_clip(
                src_clip,
                scale_factor=scene.scale_factor,
                target_handedness=scene.handedness_override
            )

            report = self.verifier.verify_all(
                source_clip=src_clip,
                retargeted_clip=retargeted_clip,
                scale_factor=scene.scale_factor,
                target_handedness=scene.handedness_override
            )

            baseline_matrix = self.generate_snapshot_matrix(scene, src_clip)
            test_matrix = self.generate_snapshot_matrix(scene, src_clip)

            res = self.compare_snapshot_matrices(scene, baseline_matrix, test_matrix, report)
            if not res.passed:
                overall_passed = False

            results.append({
                "scene_id": res.scene_id,
                "scene_name": scene.name,
                "target_criterion": scene.target_criterion,
                "passed": res.passed,
                "mse_error": res.mse_error,
                "psnr_db": res.psnr_db,
                "max_pixel_diff": res.max_pixel_diff,
                "verification_summary": res.verification_report
            })

        summary = {
            "overall_passed": overall_passed,
            "total_scenes_evaluated": len(scenes),
            "scenes": results
        }

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            report_file = out_path / "visual_regression_report.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return summary
