"""Bi-ISL Landmark Coordinate Normalization Subsystem (Prompt 19).

Normalizes 3D landmark coordinates relative to body reference points:
- Origin centered at shoulder midpoint (left_shoulder + right_shoulder) / 2
- Scale normalized by shoulder distance ||left_shoulder - right_shoulder||
- Ensures scale and position invariance for signer adaptation.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.vision.landmark_extractor import MultimodalPoseSample


class LandmarkNormalizer:
    """Normalizer converting raw frame landmarks into scale- and origin-invariant body coordinates."""

    def normalize_sample(self, sample: MultimodalPoseSample) -> MultimodalPoseSample:
        """Normalize pose, hand, and face landmarks relative to body reference points."""
        norm_sample = sample.model_copy(deep=True)

        if not sample.mask_pose or len(sample.pose_landmarks) < 13:
            return norm_sample

        pose_arr = np.array(sample.pose_landmarks, dtype=np.float32)
        left_shoulder = pose_arr[11, :3]
        right_shoulder = pose_arr[12, :3]

        shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
        shoulder_dist = float(np.linalg.norm(left_shoulder - right_shoulder))
        scale = shoulder_dist if shoulder_dist > 1e-4 else 1.0

        pose_norm = pose_arr.copy()
        pose_norm[:, :3] = (pose_arr[:, :3] - shoulder_midpoint) / scale
        norm_sample.pose_landmarks = pose_norm.tolist()

        if sample.mask_left_hand and len(sample.left_hand_landmarks) > 0:
            lh_arr = np.array(sample.left_hand_landmarks, dtype=np.float32)
            lh_arr[:, :3] = (lh_arr[:, :3] - shoulder_midpoint) / scale
            norm_sample.left_hand_landmarks = lh_arr.tolist()

        if sample.mask_right_hand and len(sample.right_hand_landmarks) > 0:
            rh_arr = np.array(sample.right_hand_landmarks, dtype=np.float32)
            rh_arr[:, :3] = (rh_arr[:, :3] - shoulder_midpoint) / scale
            norm_sample.right_hand_landmarks = rh_arr.tolist()

        if sample.mask_face and len(sample.face_landmarks) > 0:
            face_arr = np.array(sample.face_landmarks, dtype=np.float32)
            face_arr[:, :3] = (face_arr[:, :3] - shoulder_midpoint) / scale
            norm_sample.face_landmarks = face_arr.tolist()

        return norm_sample
