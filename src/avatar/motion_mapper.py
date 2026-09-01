"""Motion Sequence Mapper Interface."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseMotionMapper(ABC):
    """Abstract interface for mapping ISL IR tokens to skeletal keyframes."""
    
    @abstractmethod
    def map_to_motion_keyframes(self, isl_ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError("Interface definition only. Implementation planned for phase execution.")
