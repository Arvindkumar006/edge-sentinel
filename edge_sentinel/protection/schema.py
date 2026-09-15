from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import time

class ProtectionAction(str, Enum):
    NONE = "NONE"
    MONITORING_NOTICE = "MONITORING_NOTICE"
    WARNING_OVERLAY = "WARNING_OVERLAY"
    STRONG_OVERLAY = "STRONG_OVERLAY"
    WORKSTATION_LOCK = "WORKSTATION_LOCK"

class ShieldState(str, Enum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"

@dataclass
class ProtectionEvent:
    """Records a protection event for UI timeline and auditing."""
    timestamp: float
    time_str: str
    action: str
    reason: str

@dataclass
class ProtectionDecision:
    """Consolidated immutable decision produced by ProtectionManager."""
    shield_active: bool = False
    action: ProtectionAction = ProtectionAction.NONE
    overlay_mode: str = "NONE"            # "NONE", "WARNING", "STRONG"
    lock_triggered: bool = False
    lock_eligible: bool = False
    reasons: List[str] = field(default_factory=list)
    event_message: str = "Workstation normal"
    timestamp: float = 0.0
