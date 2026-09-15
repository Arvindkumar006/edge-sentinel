import sys
import tkinter as tk
from typing import Optional

class ScreenProtectionOverlay:
    """
    Windows-compatible desktop privacy shield overlay.
    - WARNING Mode: Amber/dark top warning banner across display.
    - STRONG Mode: Fullscreen dark privacy shield (alpha ~0.94) obscuring sensitive desktop content.
    - Strictly non-destructive: zero screenshots, zero frame saves, zero network calls.
    - Automatically hides upon threat de-escalation.
    """

    def __init__(
        self,
        master: Optional[tk.Misc] = None,
        warning_alpha: float = 0.85,
        strong_alpha: float = 0.95
    ):
        self.master = master
        self.warning_alpha = max(0.2, min(1.0, warning_alpha))
        self.strong_alpha = max(0.5, min(1.0, strong_alpha))

        self.window: Optional[tk.Toplevel] = None
        self._current_mode: str = "NONE"   # "NONE", "WARNING", "STRONG"
        self._label_title: Optional[tk.Label] = None
        self._label_body: Optional[tk.Label] = None
        self._label_sub: Optional[tk.Label] = None

    def show(self, mode: str, reason_text: str = "") -> None:
        """Activates or updates the overlay to WARNING or STRONG mode."""
        if mode not in ("WARNING", "STRONG"):
            self.hide()
            return

        if self.master is None:
            # Running in headless mode / without GUI master
            self._current_mode = mode
            return

        try:
            if self.window is None or not self.window.winfo_exists():
                self._create_window()

            self._current_mode = mode

            if mode == "STRONG":
                self._configure_strong_mode(reason_text)
            else:
                self._configure_warning_mode(reason_text)

            self.window.deiconify()
            self.window.lift()
            self.window.attributes("-topmost", True)
        except Exception as e:
            print(f"[ScreenProtectionOverlay] UI notification: {e}")
            self._current_mode = mode

    def hide(self) -> None:
        """Hides the overlay and restores normal desktop interaction."""
        self._current_mode = "NONE"
        if self.window is not None:
            try:
                if self.window.winfo_exists():
                    self.window.withdraw()
            except Exception:
                pass

    @property
    def current_mode(self) -> str:
        return self._current_mode

    @property
    def is_visible(self) -> bool:
        return self._current_mode != "NONE"

    def _create_window(self) -> None:
        self.window = tk.Toplevel(self.master)
        self.window.overrideredirect(True)  # Frameless
        self.window.attributes("-topmost", True)

        self.container = tk.Frame(self.window)
        self.container.pack(fill="both", expand=True)

        self._label_icon = tk.Label(
            self.container,
            text="⚠",
            font=("Segoe UI", 48, "bold")
        )
        self._label_icon.pack(pady=(40, 10))

        self._label_title = tk.Label(
            self.container,
            text="PRIVACY THREAT DETECTED",
            font=("Segoe UI", 24, "bold")
        )
        self._label_title.pack(pady=4)

        self._label_body = tk.Label(
            self.container,
            text="Unauthorized person or screen-capture device near workstation.",
            font=("Segoe UI", 14)
        )
        self._label_body.pack(pady=6)

        self._label_sub = tk.Label(
            self.container,
            text="Screen protection active — Normal view will automatically restore when area is secure.",
            font=("Segoe UI", 11, "italic")
        )
        self._label_sub.pack(pady=(4, 20))

    def _configure_strong_mode(self, reason_text: str) -> None:
        screen_w = self.master.winfo_screenwidth()
        screen_h = self.master.winfo_screenheight()

        self.window.geometry(f"{screen_w}x{screen_h}+0+0")
        try:
            self.window.attributes("-alpha", self.strong_alpha)
        except Exception:
            pass

        bg_color = "#09090b"  # Deep dark blackout
        accent_color = "#ef4444"  # Red
        text_color = "#f4f4f5"

        self.container.configure(bg=bg_color)
        self._label_icon.configure(bg=bg_color, fg=accent_color, text="🛡 ⚠", font=("Segoe UI", 56, "bold"))
        self._label_title.configure(bg=bg_color, fg=accent_color, text="WORKSTATION PRIVACY SHIELD ACTIVE")

        body = reason_text if reason_text else "High-risk privacy breach detected in workstation area."
        self._label_body.configure(bg=bg_color, fg=text_color, text=body)
        self._label_sub.configure(
            bg=bg_color,
            fg="#a1a1aa",
            text="Screen content obscured for privacy • Normal display will restore when threat de-escalates."
        )

    def _configure_warning_mode(self, reason_text: str) -> None:
        screen_w = self.master.winfo_screenwidth()
        banner_h = 160

        self.window.geometry(f"{screen_w}x{banner_h}+0+0")
        try:
            self.window.attributes("-alpha", self.warning_alpha)
        except Exception:
            pass

        bg_color = "#1c1917"  # Deep stone/amber
        accent_color = "#f59e0b"  # Amber
        text_color = "#fef3c7"

        self.container.configure(bg=bg_color)
        self._label_icon.configure(bg=bg_color, fg=accent_color, text="⚠", font=("Segoe UI", 28, "bold"))
        self._label_title.configure(bg=bg_color, fg=accent_color, text="PRIVACY WARNING — OBSERVATION DETECTED", font=("Segoe UI", 16, "bold"))

        body = reason_text if reason_text else "Secondary person or potential screen-capture device detected near workspace."
        self._label_body.configure(bg=bg_color, fg=text_color, text=body, font=("Segoe UI", 12))
        self._label_sub.configure(
            bg=bg_color,
            fg="#d6d3d1",
            text="Awareness mode active • Please protect confidential documents.",
            font=("Segoe UI", 10, "italic")
        )

    def destroy(self) -> None:
        if self.window is not None:
            try:
                self.window.destroy()
            except Exception:
                pass
            self.window = None
