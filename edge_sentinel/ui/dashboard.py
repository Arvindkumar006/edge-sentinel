import time
from typing import Optional
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from PIL import Image
import customtkinter as ctk

from edge_sentinel.config import AppConfig
from edge_sentinel.capture.camera import WebcamCapture
from edge_sentinel.pipeline import SentinelPipeline, PipelineOutput
from edge_sentinel.detection.visualizer import SentinelVisualizer
from edge_sentinel.context.schema import PresenceState
from edge_sentinel.threat.schema import ThreatLevel
from edge_sentinel.protection.overlay import ScreenProtectionOverlay
from edge_sentinel.protection.schema import ProtectionAction, ShieldState

class SentinelDashboard(ctk.CTk):
    """
    Phase 4 Desktop dashboard for Edge Sentinel:
    - Live video feed with workstation Privacy Zone & Device Detections
    - Active Privacy Defense card (Shield status, Protection action, Lock status)
    - Device Threat card (Phone/Camera detection, Potential screen-capture risk)
    - Threat Assessment card (Score 0-100, Level, Progress Bar)
    - Explainable Risk Reasons list
    - Protection Event Timeline
    - Primary User & Secondary People tracking metrics
    - Telemetry breakdown: Inference, Tracking, Threat, Protection, and FPS
    """

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.title(config.ui.window_title)
        self.geometry(f"{config.ui.window_width}x{config.ui.window_height}")
        self.minsize(1080, 720)

        # Pipeline components
        self.pipeline = SentinelPipeline(config)
        self.visualizer = SentinelVisualizer(
            show_fps=config.ui.display_fps,
            show_latency=config.ui.display_latency,
            show_privacy_badge=config.ui.display_privacy_badge,
            privacy_zone=self.pipeline.threat_engine.privacy_zone
        )
        self.camera = WebcamCapture(
            device_index=config.camera.device_index,
            width=config.camera.width,
            height=config.camera.height,
            target_fps=config.camera.fps,
            auto_reconnect=config.camera.auto_reconnect,
            reconnect_delay_sec=config.camera.reconnect_delay_sec
        )

        # Active Privacy Shield Overlay (Phase 4)
        self.overlay = ScreenProtectionOverlay(
            master=self,
            warning_alpha=config.protection.warning_alpha,
            strong_alpha=config.protection.strong_alpha
        )
        self.pipeline.protection_manager.set_overlay(self.overlay)

        self._is_monitoring = False
        self._running = True

        # Telemetry samples
        self._inf_latency_samples = []
        self._trk_latency_samples = []
        self._thr_latency_samples = []
        self._prot_latency_samples = []
        self._fps_samples = []

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._init_pipeline()


    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Container: Video Feed
        self.video_frame = ctk.CTkFrame(self, corner_radius=12, fg_color="#121214")
        self.video_frame.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="Initializing Edge Sentinel Phase 3...",
            font=ctk.CTkFont(family="Inter", size=16),
            fg_color="#0a0a0c",
            corner_radius=8
        )
        self.video_label.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        # Right Container: Telemetry & Controls
        self.sidebar = ctk.CTkScrollableFrame(self, corner_radius=12, fg_color="#18181b")
        self.sidebar.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)

        # 1. Header
        title_lbl = ctk.CTkLabel(
            self.sidebar,
            text="EDGE SENTINEL",
            font=ctk.CTkFont(family="Inter", size=22, weight="bold"),
            text_color="#00d2ff"
        )
        title_lbl.grid(row=0, column=0, padx=12, pady=(12, 2), sticky="w")

        subtitle_lbl = ctk.CTkLabel(
            self.sidebar,
            text="Phase 5: Qualcomm Snapdragon AI Acceleration & Active Defense",
            font=ctk.CTkFont(family="Inter", size=11),
            text_color="#a1a1aa"
        )
        subtitle_lbl.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="w")

        # 2. Main Threat Assessment Card
        self.threat_card = ctk.CTkFrame(self.sidebar, fg_color="#1a241e", border_width=1, border_color="#22c55e", corner_radius=10)
        self.threat_card.grid(row=2, column=0, padx=8, pady=4, sticky="ew")

        ctk.CTkLabel(
            self.threat_card,
            text="WORKSTATION PRIVACY THREAT",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#a1a1aa"
        ).pack(anchor="w", padx=12, pady=(8, 2))

        # Threat Score & Level Header
        threat_header = ctk.CTkFrame(self.threat_card, fg_color="transparent")
        threat_header.pack(fill="x", padx=12, pady=2)

        self.lbl_threat_level = ctk.CTkLabel(
            threat_header,
            text="SAFE",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#4ade80"
        )
        self.lbl_threat_level.pack(side="left")

        self.lbl_threat_score = ctk.CTkLabel(
            threat_header,
            text="0 / 100",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#4ade80"
        )
        self.lbl_threat_score.pack(side="right")

        # Threat Progress Bar
        self.threat_bar = ctk.CTkProgressBar(self.threat_card, height=8, corner_radius=4)
        self.threat_bar.set(0.0)
        self.threat_bar.configure(progress_color="#4ade80")
        self.threat_bar.pack(fill="x", padx=12, pady=(4, 8))

        # Threat Context Details (Zone, Approach, Persistence)
        self.lbl_zone_status = ctk.CTkLabel(
            self.threat_card,
            text="Privacy Zone: CLEAR | Approach: NONE",
            font=ctk.CTkFont(size=11),
            text_color="#d1d5db"
        )
        self.lbl_zone_status.pack(anchor="w", padx=12, pady=(0, 6))

        # 3. Active Privacy Defense Card (Phase 4)
        self.defense_card = ctk.CTkFrame(self.sidebar, fg_color="#18271e", border_width=1, border_color="#22c55e", corner_radius=8)
        self.defense_card.grid(row=3, column=0, padx=8, pady=4, sticky="ew")

        ctk.CTkLabel(
            self.defense_card,
            text="ACTIVE PRIVACY DEFENSE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#86efac"
        ).pack(anchor="w", padx=10, pady=(6, 2))

        def_row = ctk.CTkFrame(self.defense_card, fg_color="transparent")
        def_row.pack(fill="x", padx=10, pady=2)

        self.lbl_shield_status = ctk.CTkLabel(
            def_row,
            text="SHIELD: INACTIVE",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#4ade80"
        )
        self.lbl_shield_status.pack(side="left")

        self.lbl_protection_action = ctk.CTkLabel(
            def_row,
            text="ACTION: NONE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#a1a1aa"
        )
        self.lbl_protection_action.pack(side="right")

        self.lbl_defense_details = ctk.CTkLabel(
            self.defense_card,
            text="Screen protection ready • Auto-defense enabled",
            font=ctk.CTkFont(size=11),
            text_color="#dcfce7"
        )
        self.lbl_defense_details.pack(anchor="w", padx=10, pady=(0, 6))

        # 4. Device Threat Card (Phase 4)
        self.device_card = ctk.CTkFrame(self.sidebar, fg_color="#27272a", corner_radius=8)
        self.device_card.grid(row=4, column=0, padx=8, pady=4, sticky="ew")

        ctk.CTkLabel(
            self.device_card,
            text="DEVICE & SCREEN-CAPTURE THREAT",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#a1a1aa"
        ).pack(anchor="w", padx=10, pady=(6, 2))

        dev_row = ctk.CTkFrame(self.device_card, fg_color="transparent")
        dev_row.pack(fill="x", padx=10, pady=2)

        self.lbl_device_count = ctk.CTkLabel(
            dev_row,
            text="PHONE/CAM: 0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#a1a1aa"
        )
        self.lbl_device_count.pack(side="left")

        self.lbl_capture_risk = ctk.CTkLabel(
            dev_row,
            text="CAPTURE RISK: CLEAR",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#4ade80"
        )
        self.lbl_capture_risk.pack(side="right")

        self.lbl_device_details = ctk.CTkLabel(
            self.device_card,
            text="No screen-capture devices detected nearby",
            font=ctk.CTkFont(size=11),
            text_color="#71717a"
        )
        self.lbl_device_details.pack(anchor="w", padx=10, pady=(0, 6))

        # 5. Explainable Risk Reasons Card
        self.reasons_card = ctk.CTkFrame(self.sidebar, fg_color="#27272a", corner_radius=8)
        self.reasons_card.grid(row=5, column=0, padx=8, pady=4, sticky="ew")

        ctk.CTkLabel(
            self.reasons_card,
            text="EXPLAINABLE RISK SIGNALS",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#a1a1aa"
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self.lbl_reasons = ctk.CTkLabel(
            self.reasons_card,
            text="• Workstation secure — no secondary people",
            font=ctk.CTkFont(size=11),
            text_color="#d4d4d8",
            justify="left",
            wraplength=340
        )
        self.lbl_reasons.pack(anchor="w", padx=10, pady=(0, 8))

        # 6. Protection Event Timeline Card (Phase 4)
        self.timeline_card = ctk.CTkFrame(self.sidebar, fg_color="#27272a", corner_radius=8)
        self.timeline_card.grid(row=6, column=0, padx=8, pady=4, sticky="ew")

        ctk.CTkLabel(
            self.timeline_card,
            text="PROTECTION EVENT TIMELINE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#a1a1aa"
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self.lbl_timeline = ctk.CTkLabel(
            self.timeline_card,
            text="No protection events recorded yet",
            font=ctk.CTkFont(size=10),
            text_color="#a1a1aa",
            justify="left",
            wraplength=340
        )
        self.lbl_timeline.pack(anchor="w", padx=10, pady=(0, 8))

        # 7. Primary User Presence Card
        self.presence_card = ctk.CTkFrame(self.sidebar, fg_color="#1e2922", border_width=1, border_color="#22c55e", corner_radius=8)
        self.presence_card.grid(row=7, column=0, padx=8, pady=4, sticky="ew")

        ctk.CTkLabel(
            self.presence_card,
            text="PRIMARY WORKSPACE USER",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#86efac"
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self.lbl_primary_status = ctk.CTkLabel(
            self.presence_card,
            text="PRESENT (ACTIVE)",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#4ade80"
        )
        self.lbl_primary_status.pack(anchor="w", padx=10, pady=(0, 2))

        self.lbl_primary_details = ctk.CTkLabel(
            self.presence_card,
            text="Track ID: #1 | Visible: 0.0s",
            font=ctk.CTkFont(size=11),
            text_color="#dcfce7"
        )
        self.lbl_primary_details.pack(anchor="w", padx=10, pady=(0, 6))

        # 8. Secondary People Card
        self.secondary_card = ctk.CTkFrame(self.sidebar, fg_color="#27272a", corner_radius=8)
        self.secondary_card.grid(row=8, column=0, padx=8, pady=4, sticky="ew")

        ctk.CTkLabel(
            self.secondary_card,
            text="SECONDARY PEOPLE DETECTION",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#a1a1aa"
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self.lbl_secondary_status = ctk.CTkLabel(
            self.secondary_card,
            text="NONE (Clear)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#a1a1aa"
        )
        self.lbl_secondary_status.pack(anchor="w", padx=10, pady=(0, 2))

        self.lbl_secondary_details = ctk.CTkLabel(
            self.secondary_card,
            text="No secondary persons nearby",
            font=ctk.CTkFont(size=11),
            text_color="#71717a"
        )
        self.lbl_secondary_details.pack(anchor="w", padx=10, pady=(0, 6))

        # 9. People & Active Tracks
        counts_frame = ctk.CTkFrame(self.sidebar, fg_color="#27272a", corner_radius=8)
        counts_frame.grid(row=9, column=0, padx=8, pady=4, sticky="ew")
        counts_frame.grid_columnconfigure((0, 1), weight=1)

        pcount_box = ctk.CTkFrame(counts_frame, fg_color="#18181b", corner_radius=6)
        pcount_box.grid(row=0, column=0, padx=(6, 3), pady=6, sticky="ew")
        ctk.CTkLabel(pcount_box, text="PEOPLE IN VIEW", font=ctk.CTkFont(size=9, weight="bold"), text_color="#71717a").pack(pady=(4, 0))
        self.lbl_people_count = ctk.CTkLabel(pcount_box, text="0", font=ctk.CTkFont(size=18, weight="bold"), text_color="#38bdf8")
        self.lbl_people_count.pack(pady=(0, 4))

        tids_box = ctk.CTkFrame(counts_frame, fg_color="#18181b", corner_radius=6)
        tids_box.grid(row=0, column=1, padx=(3, 6), pady=6, sticky="ew")
        ctk.CTkLabel(tids_box, text="ACTIVE TRACKS", font=ctk.CTkFont(size=9, weight="bold"), text_color="#71717a").pack(pady=(4, 0))
        self.lbl_active_tracks = ctk.CTkLabel(tids_box, text="None", font=ctk.CTkFont(size=12, weight="bold"), text_color="#facc15")
        self.lbl_active_tracks.pack(pady=(0, 4))

        # 10. AI Acceleration Status Card (Phase 5)
        self.ai_card = ctk.CTkFrame(self.sidebar, fg_color="#18271e", border_width=1, border_color="#22c55e", corner_radius=8)
        self.ai_card.grid(row=10, column=0, padx=8, pady=4, sticky="ew")

        ctk.CTkLabel(
            self.ai_card,
            text="AI ACCELERATION",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#86efac"
        ).pack(anchor="w", padx=10, pady=(6, 2))

        ai_row1 = ctk.CTkFrame(self.ai_card, fg_color="transparent")
        ai_row1.pack(fill="x", padx=10, pady=2)

        self.lbl_ai_backend = ctk.CTkLabel(
            ai_row1,
            text="Backend: CPU",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#dcfce7"
        )
        self.lbl_ai_backend.pack(side="left")

        self.lbl_ai_status = ctk.CTkLabel(
            ai_row1,
            text="ACTIVE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#4ade80"
        )
        self.lbl_ai_status.pack(side="right")

        ai_row2 = ctk.CTkFrame(self.ai_card, fg_color="transparent")
        ai_row2.pack(fill="x", padx=10, pady=2)

        self.lbl_ai_accelerator = ctk.CTkLabel(
            ai_row2,
            text="Accelerator: CPU",
            font=ctk.CTkFont(size=11),
            text_color="#a1a1aa"
        )
        self.lbl_ai_accelerator.pack(side="left")

        self.lbl_ai_model = ctk.CTkLabel(
            ai_row2,
            text="Model: YOLOv8n",
            font=ctk.CTkFont(size=11),
            text_color="#a1a1aa"
        )
        self.lbl_ai_model.pack(side="right")

        self.lbl_ai_notice = ctk.CTkLabel(
            self.ai_card,
            text="Local PyTorch inference",
            font=ctk.CTkFont(size=10),
            text_color="#71717a",
            wraplength=340,
            justify="left"
        )
        self.lbl_ai_notice.pack(anchor="w", padx=10, pady=(0, 6))

        # 11. Performance Telemetry (FPS, Inference, Tracking, Threat, Protection ms)
        telemetry_frame = ctk.CTkFrame(self.sidebar, fg_color="#27272a", corner_radius=8)
        telemetry_frame.grid(row=11, column=0, padx=8, pady=4, sticky="ew")
        telemetry_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        # FPS
        f_box = ctk.CTkFrame(telemetry_frame, fg_color="#18181b", corner_radius=6)
        f_box.grid(row=0, column=0, padx=(4, 2), pady=6, sticky="ew")
        ctk.CTkLabel(f_box, text="FPS", font=ctk.CTkFont(size=8, weight="bold"), text_color="#71717a").pack(pady=(3, 0))
        self.lbl_fps = ctk.CTkLabel(f_box, text="--", font=ctk.CTkFont(size=11, weight="bold"), text_color="#4ade80")
        self.lbl_fps.pack(pady=(0, 3))

        # Inf ms
        i_box = ctk.CTkFrame(telemetry_frame, fg_color="#18181b", corner_radius=6)
        i_box.grid(row=0, column=1, padx=2, pady=6, sticky="ew")
        ctk.CTkLabel(i_box, text="INF", font=ctk.CTkFont(size=8, weight="bold"), text_color="#71717a").pack(pady=(3, 0))
        self.lbl_inf = ctk.CTkLabel(i_box, text="-- ms", font=ctk.CTkFont(size=11, weight="bold"), text_color="#facc15")
        self.lbl_inf.pack(pady=(0, 3))

        # Trk ms
        t_box = ctk.CTkFrame(telemetry_frame, fg_color="#18181b", corner_radius=6)
        t_box.grid(row=0, column=2, padx=2, pady=6, sticky="ew")
        ctk.CTkLabel(t_box, text="TRK", font=ctk.CTkFont(size=8, weight="bold"), text_color="#71717a").pack(pady=(3, 0))
        self.lbl_trk = ctk.CTkLabel(t_box, text="-- ms", font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8")
        self.lbl_trk.pack(pady=(0, 3))

        # Thr ms
        th_box = ctk.CTkFrame(telemetry_frame, fg_color="#18181b", corner_radius=6)
        th_box.grid(row=0, column=3, padx=2, pady=6, sticky="ew")
        ctk.CTkLabel(th_box, text="THR", font=ctk.CTkFont(size=8, weight="bold"), text_color="#71717a").pack(pady=(3, 0))
        self.lbl_thr = ctk.CTkLabel(th_box, text="-- ms", font=ctk.CTkFont(size=11, weight="bold"), text_color="#c084fc")
        self.lbl_thr.pack(pady=(0, 3))

        # Prot ms
        pr_box = ctk.CTkFrame(telemetry_frame, fg_color="#18181b", corner_radius=6)
        pr_box.grid(row=0, column=4, padx=(2, 4), pady=6, sticky="ew")
        ctk.CTkLabel(pr_box, text="PROT", font=ctk.CTkFont(size=8, weight="bold"), text_color="#71717a").pack(pady=(3, 0))
        self.lbl_prot = ctk.CTkLabel(pr_box, text="-- ms", font=ctk.CTkFont(size=11, weight="bold"), text_color="#34d399")
        self.lbl_prot.pack(pady=(0, 3))

        # 12. Interactive Controls
        ctrl_card = ctk.CTkFrame(self.sidebar, fg_color="#27272a", corner_radius=8)
        ctrl_card.grid(row=12, column=0, padx=8, pady=4, sticky="ew")


        ctk.CTkLabel(ctrl_card, text="CONTROLS", font=ctk.CTkFont(size=10, weight="bold"), text_color="#71717a").pack(anchor="w", padx=10, pady=(6, 2))

        self.slider_lbl = ctk.CTkLabel(
            ctrl_card,
            text=f"Detection Confidence: {int(self.config.detection.confidence_threshold * 100)}%",
            font=ctk.CTkFont(size=11)
        )
        self.slider_lbl.pack(anchor="w", padx=10, pady=(2, 0))

        self.conf_slider = ctk.CTkSlider(
            ctrl_card,
            from_=0.15,
            to=0.90,
            number_of_steps=15,
            command=self._on_confidence_slider_change
        )
        self.conf_slider.set(self.config.detection.confidence_threshold)
        self.conf_slider.pack(fill="x", padx=10, pady=(2, 6))

        self.btn_toggle = ctk.CTkButton(
            ctrl_card,
            text="Pause Monitoring",
            fg_color="#dc2626",
            hover_color="#b91c1c",
            font=ctk.CTkFont(weight="bold"),
            command=self._toggle_monitoring
        )
        self.btn_toggle.pack(fill="x", padx=10, pady=(2, 4))

        self.btn_reconnect = ctk.CTkButton(
            ctrl_card,
            text="Reconnect Camera",
            fg_color="#3f3f46",
            hover_color="#52525b",
            command=self._reconnect_camera
        )
        self.btn_reconnect.pack(fill="x", padx=10, pady=(0, 6))

    def _init_pipeline(self):
        loaded = self.pipeline.load()
        if not loaded:
            self.video_label.configure(text="Failed to load YOLO model.")
            return

        cam_started = self.camera.start()
        self._is_monitoring = True
        self.after(30, self._process_loop)

    def _on_confidence_slider_change(self, value: float):
        self.pipeline.set_confidence_threshold(value)
        self.slider_lbl.configure(text=f"Detection Confidence: {int(value * 100)}%")

    def _toggle_monitoring(self):
        self._is_monitoring = not self._is_monitoring
        if self._is_monitoring:
            self.btn_toggle.configure(text="Pause Monitoring", fg_color="#dc2626", hover_color="#b91c1c")
        else:
            self.btn_toggle.configure(text="Resume Monitoring", fg_color="#16a34a", hover_color="#15803d")

    def _reconnect_camera(self):
        self.camera.stop()
        time.sleep(0.3)
        self.pipeline.reset_state()
        self.camera.start()

    def _process_loop(self):
        if not self._running:
            return

        if self._is_monitoring:
            ret, frame = self.camera.read()

            if ret and frame is not None:
                output: PipelineOutput = self.pipeline.process_frame(frame)

                self._inf_latency_samples.append(output.inference_latency_ms)
                self._trk_latency_samples.append(output.tracking_overhead_ms)
                self._thr_latency_samples.append(output.threat_overhead_ms)
                self._prot_latency_samples.append(output.protection_overhead_ms)
                if output.fps > 0:
                    self._fps_samples.append(output.fps)

                vis_frame = self.visualizer.render(
                    frame=output.frame,
                    tracks=output.tracks,
                    context=output.context,
                    threat=output.threat,
                    inference_latency_ms=output.inference_latency_ms,
                    tracking_overhead_ms=output.tracking_overhead_ms,
                    threat_overhead_ms=output.threat_overhead_ms,
                    fps=output.fps,
                    device_name=output.device_name,
                    protection=output.protection,
                    protection_overhead_ms=output.protection_overhead_ms,
                    device_detections=output.device_detections
                )

                rgb_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb_frame.shape

                target_w = max(480, self.video_frame.winfo_width() - 24)
                target_h = max(360, self.video_frame.winfo_height() - 24)
                scale = min(target_w / w, target_h / h)
                disp_w = max(1, int(w * scale))
                disp_h = max(1, int(h * scale))

                pil_img = Image.fromarray(rgb_frame).resize((disp_w, disp_h), Image.Resampling.BILINEAR)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))

                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img

                # Update UI from context, threat, & protection
                ctx = output.context
                thr = output.threat
                prot = output.protection

                # 1. Threat Card UI
                self.threat_bar.set(thr.score / 100.0)
                self.lbl_threat_score.configure(text=f"{thr.score:.0f} / 100")

                if thr.level == ThreatLevel.CRITICAL.value:
                    col = "#ef4444"
                    self.threat_card.configure(fg_color="#361a1a", border_color="#ef4444")
                elif thr.level == ThreatLevel.WARNING.value:
                    col = "#f97316"
                    self.threat_card.configure(fg_color="#302014", border_color="#f97316")
                elif thr.level == ThreatLevel.MONITORING.value:
                    col = "#eab308"
                    self.threat_card.configure(fg_color="#2b2314", border_color="#eab308")
                else:
                    col = "#22c55e"
                    self.threat_card.configure(fg_color="#18271e", border_color="#22c55e")

                self.lbl_threat_level.configure(text=thr.level, text_color=col)
                self.lbl_threat_score.configure(text_color=col)
                self.threat_bar.configure(progress_color=col)

                zone_txt = "BREACH" if thr.inside_privacy_zone else ("APPROACHING" if thr.approaching_privacy_zone else "CLEAR")
                appr_txt = f"{thr.approach_velocity:.0f} px/s" if thr.approaching_privacy_zone else "NONE"
                self.lbl_zone_status.configure(text=f"Zone: {zone_txt} | Approach: {appr_txt} | Visible: {thr.duration:.1f}s")

                # 2. Active Privacy Defense Card UI (Phase 4)
                if prot is not None:
                    if prot.shield_active:
                        shield_col = "#ef4444" if prot.action in (ProtectionAction.STRONG_OVERLAY, ProtectionAction.WORKSTATION_LOCK) else "#f97316"
                        card_bg = "#361a1a" if prot.action in (ProtectionAction.STRONG_OVERLAY, ProtectionAction.WORKSTATION_LOCK) else "#302014"
                        self.defense_card.configure(fg_color=card_bg, border_color=shield_col)
                        self.lbl_shield_status.configure(text="SHIELD: ACTIVE", text_color=shield_col)
                    else:
                        self.defense_card.configure(fg_color="#18271e", border_color="#22c55e")
                        self.lbl_shield_status.configure(text="SHIELD: INACTIVE", text_color="#4ade80")

                    self.lbl_protection_action.configure(text=f"ACTION: {prot.action.value}")
                    self.lbl_defense_details.configure(text=prot.event_message)

                # 3. Device Threat Card UI (Phase 4)
                dev_count = len(output.device_detections) if output.device_detections else 0
                dev_label = f"PHONE/CAM: {dev_count}" + (" (Cell Phone)" if dev_count > 0 else "")
                self.lbl_device_count.configure(text=dev_label)

                if getattr(thr, "potential_screen_capture_risk", False):
                    self.lbl_capture_risk.configure(text="⚠️ RISK DETECTED", text_color="#ef4444")
                    self.lbl_device_details.configure(text="Potential screen-capture risk near workstation!", text_color="#fca5a5")
                    self.device_card.configure(fg_color="#361a1a", border_width=1, border_color="#ef4444")
                elif dev_count > 0:
                    self.lbl_capture_risk.configure(text="DEVICE DETECTED", text_color="#facc15")
                    self.lbl_device_details.configure(text="Smartphone observed in field of view", text_color="#fef08a")
                    self.device_card.configure(fg_color="#2b2314", border_width=1, border_color="#eab308")
                else:
                    self.lbl_capture_risk.configure(text="CAPTURE RISK: CLEAR", text_color="#4ade80")
                    self.lbl_device_details.configure(text="No screen-capture devices detected nearby", text_color="#71717a")
                    self.device_card.configure(fg_color="#27272a", border_width=0)

                # 4. Risk Reasons
                reasons_str = "\n".join(f"• {r}" for r in thr.reasons[:4])
                self.lbl_reasons.configure(text=reasons_str)

                # 5. Protection Event Timeline (Phase 4)
                evts = self.pipeline.protection_manager.timeline
                if evts:
                    timeline_str = "\n".join(f"• [{e.time_str}] {e.action} — {e.reason}" for e in evts[:3])
                    self.lbl_timeline.configure(text=timeline_str)
                else:
                    self.lbl_timeline.configure(text="No protection events recorded yet")

                # 6. Primary User Presence UI
                if ctx.presence_state == PresenceState.ACTIVE.value:
                    self.presence_card.configure(fg_color="#18271e", border_color="#22c55e")
                    self.lbl_primary_status.configure(text="PRESENT (ACTIVE)", text_color="#4ade80")
                    self.lbl_primary_details.configure(
                        text=f"Track ID: #{ctx.primary_track_id} | Visible: {ctx.primary_presence_duration:.1f}s",
                        text_color="#dcfce7"
                    )
                elif ctx.presence_state == PresenceState.TEMPORARILY_ABSENT.value:
                    self.presence_card.configure(fg_color="#2b2314", border_color="#eab308")
                    self.lbl_primary_status.configure(text="TEMPORARILY ABSENT", text_color="#facc15")
                    self.lbl_primary_details.configure(
                        text=f"Absent: {ctx.user_absent_duration:.1f}s | Grace Period",
                        text_color="#fef08a"
                    )
                else:
                    self.presence_card.configure(fg_color="#27272a", border_color="#52525b")
                    self.lbl_primary_status.configure(text="ABSENT", text_color="#a1a1aa")
                    self.lbl_primary_details.configure(
                        text=f"Absent: {ctx.user_absent_duration:.1f}s | User Away",
                        text_color="#71717a"
                    )

                # 7. Secondary People UI
                if ctx.secondary_people_count > 0:
                    self.secondary_card.configure(fg_color="#361a1a", border_width=1, border_color="#f87171")
                    self.lbl_secondary_status.configure(text=f"⚠️ {ctx.secondary_people_count} DETECTED", text_color="#f87171")
                    prox_str = f"Prox: {ctx.nearest_secondary_distance:.2f}" if ctx.nearest_secondary_distance is not None else ""
                    self.lbl_secondary_details.configure(
                        text=f"Visible: {ctx.secondary_presence_duration:.1f}s | {prox_str}",
                        text_color="#fca5a5"
                    )
                else:
                    self.secondary_card.configure(fg_color="#27272a", border_width=0)
                    self.lbl_secondary_status.configure(text="NONE (Clear)", text_color="#a1a1aa")
                    self.lbl_secondary_details.configure(text="No secondary persons nearby", text_color="#71717a")

                # 8. People & Tracks
                self.lbl_people_count.configure(text=str(ctx.people_count))
                tids_str = ", ".join(f"#{tid}" for tid in ctx.active_track_ids) if ctx.active_track_ids else "None"
                self.lbl_active_tracks.configure(text=tids_str)

                # 9. AI Acceleration Status Card UI (Phase 5)
                b_type = getattr(output, "backend_type", "CPU")
                acc_type = getattr(output, "accelerator_type", "CPU")
                b_status = getattr(output, "backend_status", "ACTIVE")
                b_model = getattr(output, "model_name", "YOLOv8n")
                notice = getattr(output, "fallback_notice", None)

                self.lbl_ai_backend.configure(text=f"Backend: {b_type}")
                self.lbl_ai_accelerator.configure(text=f"Accelerator: {acc_type}")
                self.lbl_ai_model.configure(text=f"Model: {b_model}")

                if acc_type == "NPU":
                    self.ai_card.configure(fg_color="#18271e", border_color="#22c55e")
                    self.lbl_ai_status.configure(text="ACTIVE (NPU)", text_color="#4ade80")
                    self.lbl_ai_notice.configure(
                        text="Qualcomm Hexagon NPU verified active",
                        text_color="#86efac"
                    )
                elif b_status == "FALLBACK":
                    self.ai_card.configure(fg_color="#302014", border_color="#f59e0b")
                    self.lbl_ai_status.configure(text="FALLBACK", text_color="#f59e0b")
                    fallback_text = notice or "Snapdragon backend unavailable — CPU fallback active"
                    self.lbl_ai_notice.configure(
                        text=fallback_text,
                        text_color="#fbbf24"
                    )
                else:
                    self.ai_card.configure(fg_color="#1f242e", border_color="#3b82f6")
                    self.lbl_ai_status.configure(text="ACTIVE", text_color="#60a5fa")
                    self.lbl_ai_notice.configure(
                        text="Host CPU execution active",
                        text_color="#93c5fd"
                    )

                # 10. Telemetry breakdown
                self.lbl_fps.configure(text=f"{output.fps:.1f}")
                self.lbl_inf.configure(text=f"{output.inference_latency_ms:.1f} ms")
                self.lbl_trk.configure(text=f"{output.tracking_overhead_ms:.2f} ms")
                self.lbl_thr.configure(text=f"{output.threat_overhead_ms:.2f} ms")
                self.lbl_prot.configure(text=f"{output.protection_overhead_ms:.2f} ms")

            else:
                if not self.camera.is_opened:
                    self.video_label.configure(image=None, text="Camera Offline. Click Reconnect Camera to retry.")

        self.after(33, self._process_loop)

    def on_close(self):
        self._running = False
        self._is_monitoring = False
        self.camera.stop()
        if hasattr(self, "overlay") and self.overlay is not None:
            self.overlay.destroy()
        self.destroy()

