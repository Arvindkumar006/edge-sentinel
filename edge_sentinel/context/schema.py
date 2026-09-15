from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List

class PresenceState(str, Enum):
    ACTIVE = "ACTIVE"
    TEMPORARILY_ABSENT = "TEMPORARILY_ABSENT"
    ABSENT = "ABSENT"

@dataclass
class WorkspaceContext:
    """
    Structured context of workstation presence and surrounding people.
    Consumed by the UI, alerting, and downstream privacy modules.
    """
    primary_user_present: bool
    primary_track_id: Optional[int]
    presence_state: str                    # "ACTIVE", "TEMPORARILY_ABSENT", "ABSENT"
    people_count: int
    secondary_people_count: int
    nearest_secondary_distance: Optional[float]
    secondary_presence_duration: float
    user_absent_duration: float
    primary_presence_duration: float
    active_track_ids: List[int] = field(default_factory=list)
    timestamp: float = 0.0
