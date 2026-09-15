import sys
import time
from typing import Optional

class WorkstationLocker:
    """
    Native Windows workstation locker with strict guardrails.
    - Disabled by default
    - Only triggered when explicitly enabled, threat is CRITICAL, and user is ABSENT
    - Enforces cooldown and single-trigger prevention to avoid repeatedly locking
    - Strictly uses standard OS LockWorkStation; zero custom authentication or credentials
    """

    def __init__(
        self,
        enabled: bool = False,
        cooldown_sec: float = 60.0,
        dry_run: bool = False
    ):
        self.enabled = enabled
        self.cooldown_sec = cooldown_sec
        self.dry_run = dry_run

        self.last_lock_time: Optional[float] = None
        self.lock_count: int = 0
        self.was_triggered: bool = False

    def is_eligible(self, is_critical: bool, is_user_absent: bool, now: Optional[float] = None) -> bool:
        """
        Determines whether the system qualifies for a workstation lock.
        Requirements:
        1. Threat level must be CRITICAL
        2. Primary user must be ABSENT
        3. Cooldown duration since last lock must have elapsed
        """
        if not is_critical or not is_user_absent:
            return False

        t = now if now is not None else time.perf_counter()
        if self.last_lock_time is not None:
            if (t - self.last_lock_time) < self.cooldown_sec:
                return False

        return True

    def trigger_lock(self, is_critical: bool, is_user_absent: bool, now: Optional[float] = None) -> bool:
        """
        Executes workstation lock if eligible and enabled.
        Returns True if a lock action was actually performed.
        """
        t = now if now is not None else time.perf_counter()

        if not self.is_eligible(is_critical, is_user_absent, now=t):
            return False

        if not self.enabled:
            # Policy is met, but locking is disabled in config
            return False

        self.last_lock_time = t
        self.lock_count += 1
        self.was_triggered = True

        if self.dry_run:
            print("[WorkstationLocker] Dry run: LockWorkStation qualified and recorded.")
            return True

        if sys.platform == "win32":
            try:
                import ctypes
                result = ctypes.windll.user32.LockWorkStation()
                return bool(result)
            except Exception as e:
                print(f"[WorkstationLocker] Error calling LockWorkStation: {e}")
                return False
        else:
            print("[WorkstationLocker] LockWorkStation only supported on Windows.")
            return False

    def reset(self) -> None:
        self.last_lock_time = None
        self.lock_count = 0
        self.was_triggered = False
