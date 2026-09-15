from edge_sentinel.protection.schema import (
    ProtectionAction,
    ShieldState,
    ProtectionDecision,
    ProtectionEvent
)
from edge_sentinel.protection.lock import WorkstationLocker
from edge_sentinel.protection.overlay import ScreenProtectionOverlay
from edge_sentinel.protection.manager import ProtectionManager

__all__ = [
    "ProtectionAction",
    "ShieldState",
    "ProtectionDecision",
    "ProtectionEvent",
    "WorkstationLocker",
    "ScreenProtectionOverlay",
    "ProtectionManager"
]
