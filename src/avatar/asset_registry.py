"""Avatar Sign Asset Registry Subsystem (Prompt 72).

Manages 3D avatar sign motion assets, storing required metadata fields and
enforcing explicit validation status tagging.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


class ValidationStatus(str, Enum):
    """Validation status for 3D avatar sign assets."""
    UNVALIDATED = "UNVALIDATED"
    VALIDATED = "VALIDATED"
    PENDING_REVIEW = "PENDING_REVIEW"
    REJECTED = "REJECTED"


@dataclass
class SignAssetMetadata:
    """Metadata container for a 3D avatar sign asset."""
    sign_id: str
    gloss: str
    animation_asset: str
    duration: float  # in seconds
    dominant_hand: str  # "RIGHT", "LEFT", "BOTH"
    start_pose: Dict[str, Any] = field(default_factory=dict)
    end_pose: Dict[str, Any] = field(default_factory=dict)
    required_facial_markers: List[str] = field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    reviewer: Optional[str] = None
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Validate fields upon initialization."""
        if not self.sign_id or not isinstance(self.sign_id, str):
            raise ValueError("sign_id must be a non-empty string.")
        if not self.gloss or not isinstance(self.gloss, str):
            raise ValueError("gloss must be a non-empty string.")
        if not self.animation_asset or not isinstance(self.animation_asset, str):
            raise ValueError("animation_asset must be a non-empty string.")
        if self.duration <= 0:
            raise ValueError("duration must be positive.")
        
        valid_hands = {"RIGHT", "LEFT", "BOTH"}
        if self.dominant_hand.upper() not in valid_hands:
            raise ValueError(f"dominant_hand must be one of {valid_hands}, got {self.dominant_hand}")
        self.dominant_hand = self.dominant_hand.upper()

        if isinstance(self.validation_status, str):
            self.validation_status = ValidationStatus(self.validation_status)

    @property
    def is_validated(self) -> bool:
        """Return True if asset has been officially validated."""
        return self.validation_status == ValidationStatus.VALIDATED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize asset metadata to dictionary."""
        d = asdict(self)
        d["validation_status"] = self.validation_status.value
        d["is_validated"] = self.is_validated
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignAssetMetadata":
        """Construct SignAssetMetadata from dictionary."""
        data_copy = dict(data)
        data_copy.pop("is_validated", None)
        if "validation_status" in data_copy:
            data_copy["validation_status"] = ValidationStatus(data_copy["validation_status"])
        return cls(**data_copy)


class AvatarSignAssetRegistry:
    """Registry managing 3D avatar sign motion assets."""

    def __init__(self) -> None:
        self._assets: Dict[str, SignAssetMetadata] = {}

    def register_asset(self, asset: SignAssetMetadata) -> None:
        """Register a new sign asset or overwrite an existing one."""
        if not isinstance(asset, SignAssetMetadata):
            raise TypeError("asset must be an instance of SignAssetMetadata.")
        self._assets[asset.sign_id] = asset

    def get_asset(self, sign_id: str) -> Optional[SignAssetMetadata]:
        """Retrieve asset metadata by sign_id."""
        return self._assets.get(sign_id)

    def get_asset_by_gloss(self, gloss: str) -> Optional[SignAssetMetadata]:
        """Retrieve asset metadata by ISL gloss token."""
        gloss_upper = gloss.upper()
        for asset in self._assets.values():
            if asset.gloss.upper() == gloss_upper:
                return asset
        return None

    def list_assets(
        self, validation_status: Optional[Union[ValidationStatus, str]] = None
    ) -> List[SignAssetMetadata]:
        """List all registered assets, optionally filtering by validation status."""
        if validation_status is None:
            return list(self._assets.values())
        
        target_status = (
            ValidationStatus(validation_status)
            if isinstance(validation_status, str)
            else validation_status
        )
        return [
            asset for asset in self._assets.values()
            if asset.validation_status == target_status
        ]

    def get_unvalidated_assets(self) -> List[SignAssetMetadata]:
        """Retrieve all assets clearly marked as UNVALIDATED."""
        return self.list_assets(ValidationStatus.UNVALIDATED)

    def get_validated_assets(self) -> List[SignAssetMetadata]:
        """Retrieve all assets marked as VALIDATED."""
        return self.list_assets(ValidationStatus.VALIDATED)

    def validate_asset(
        self, sign_id: str, reviewer: str, version: Optional[str] = None
    ) -> SignAssetMetadata:
        """Mark an unvalidated or pending asset as officially VALIDATED by a reviewer."""
        asset = self.get_asset(sign_id)
        if asset is None:
            raise KeyError(f"Sign asset '{sign_id}' not found in registry.")
        if not reviewer or not isinstance(reviewer, str):
            raise ValueError("Reviewer must be a non-empty string.")

        asset.validation_status = ValidationStatus.VALIDATED
        asset.reviewer = reviewer
        if version:
            asset.version = version
        return asset

    def unvalidate_asset(self, sign_id: str, reason: Optional[str] = None) -> SignAssetMetadata:
        """Revert an asset's validation status to UNVALIDATED."""
        asset = self.get_asset(sign_id)
        if asset is None:
            raise KeyError(f"Sign asset '{sign_id}' not found in registry.")
        asset.validation_status = ValidationStatus.UNVALIDATED
        return asset

    def export_registry(self) -> List[Dict[str, Any]]:
        """Export all registered assets as a list of dictionaries."""
        return [asset.to_dict() for asset in self._assets.values()]

    def load_registry(self, assets_data: List[Dict[str, Any]]) -> None:
        """Load multiple assets from a list of dictionaries."""
        for item in assets_data:
            asset = SignAssetMetadata.from_dict(item)
            self.register_asset(asset)

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Save entire asset catalog to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export_registry(), f, indent=2)

    def load_json(self, file_path: Union[str, Path]) -> None:
        """Load asset catalog from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON registry file not found at: {file_path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_registry(data)

    @classmethod
    def create_seeded_registry(cls) -> "AvatarSignAssetRegistry":
        """Create a registry pre-seeded with baseline domain sign assets."""
        registry = cls()
        seed_data = [
            {
                "sign_id": "SIGN_DOCTOR_01",
                "gloss": "DOCTOR",
                "animation_asset": "assets/motions/doctor.anim",
                "duration": 1.25,
                "dominant_hand": "RIGHT",
                "start_pose": {"wrist_pos": [0.15, 1.2, 0.4], "handshape": "FLAT_HAND"},
                "end_pose": {"wrist_pos": [0.10, 1.2, 0.35], "handshape": "FLAT_HAND"},
                "required_facial_markers": ["browDownLeft", "browDownRight"],
                "validation_status": "VALIDATED",
                "reviewer": "DHH_Linguist_01",
                "version": "1.0.0"
            },
            {
                "sign_id": "SIGN_FEVER_01",
                "gloss": "FEVER",
                "animation_asset": "assets/motions/fever.anim",
                "duration": 1.10,
                "dominant_hand": "RIGHT",
                "start_pose": {"wrist_pos": [0.0, 1.45, 0.25], "handshape": "FLAT_HAND"},
                "end_pose": {"wrist_pos": [0.0, 1.45, 0.20], "handshape": "FLAT_HAND"},
                "required_facial_markers": ["eyeSquintLeft", "eyeSquintRight"],
                "validation_status": "UNVALIDATED",
                "reviewer": None,
                "version": "0.9.0"
            },
            {
                "sign_id": "SIGN_MEDICINE_01",
                "gloss": "MEDICINE",
                "animation_asset": "assets/motions/medicine.anim",
                "duration": 1.40,
                "dominant_hand": "BOTH",
                "start_pose": {"wrist_pos_R": [0.12, 1.1, 0.3], "wrist_pos_L": [-0.12, 1.1, 0.3]},
                "end_pose": {"wrist_pos_R": [0.05, 1.1, 0.3], "wrist_pos_L": [-0.05, 1.1, 0.3]},
                "required_facial_markers": [],
                "validation_status": "VALIDATED",
                "reviewer": "DHH_Linguist_02",
                "version": "1.0.0"
            },
            {
                "sign_id": "SIGN_TIMING_01",
                "gloss": "TIME",
                "animation_asset": "assets/motions/time.anim",
                "duration": 0.95,
                "dominant_hand": "RIGHT",
                "start_pose": {"wrist_pos": [0.10, 1.0, 0.3], "handshape": "INDEX_POINT"},
                "end_pose": {"wrist_pos": [-0.10, 1.0, 0.3], "handshape": "INDEX_POINT"},
                "required_facial_markers": [],
                "validation_status": "UNVALIDATED",
                "reviewer": None,
                "version": "0.8.5"
            },
            {
                "sign_id": "SIGN_HELP_01",
                "gloss": "HELP",
                "animation_asset": "assets/motions/help.anim",
                "duration": 1.30,
                "dominant_hand": "BOTH",
                "start_pose": {"wrist_pos_R": [0.0, 1.0, 0.35], "wrist_pos_L": [0.0, 0.95, 0.35]},
                "end_pose": {"wrist_pos_R": [0.0, 1.25, 0.35], "wrist_pos_L": [0.0, 1.20, 0.35]},
                "required_facial_markers": ["browOuterUpLeft", "browOuterUpRight"],
                "validation_status": "VALIDATED",
                "reviewer": "DHH_Linguist_01",
                "version": "1.0.0"
            }
        ]
        registry.load_registry(seed_data)
        return registry
