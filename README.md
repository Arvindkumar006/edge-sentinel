# Edge Sentinel — On-Device AI Privacy Agent (Phase 5: Qualcomm Snapdragon AI Acceleration)

Edge Sentinel is a privacy-preserving AI security desktop application designed to monitor workstations for unauthorized onlookers, shoulder surfing, and physical privacy risks using lightweight computer vision models running 100% locally.

The application features a pluggable inference backend abstraction layer:
```text
InferenceBackend
├── CPUBackend (Ultralytics PyTorch Local)
└── SnapdragonBackend (Qualcomm QNN / Hexagon NPU)
```
The pipeline automatically selects the best available backend (`auto`, `snapdragon`, or `cpu`) and seamlessly falls back to CPU if Snapdragon NPU hardware or runtime is unavailable, guaranteeing zero application crashes.

---

## Architecture Pipeline (Phase 5)

```text
[Webcam (OpenCV Thread-Safe DirectShow)]
        │
        ▼
[Inference Layer (BaseInferenceEngine Abstraction)]
  ├── CPUBackend: YOLOv8n (PyTorch CPU fallback)
  └── SnapdragonBackend: Qualcomm QNN (Hexagon NPU)
        │
        ▼
[Person & Device Detector]
  ├── Person detection (COCO class 0)
  └── Screen-capture device detection (COCO class 67: cell phone)
        │
        ▼
[PersonTracker (Centroid & IoU Association)]
  └── Assigns persistent track IDs, trajectory history, and proximity
        │
        ▼
[PrimaryUserEstimator & WorkspaceContext State Machine]
  ├── ACTIVE ⇆ TEMPORARILY_ABSENT ⇆ ABSENT
  └── Spatial/temporal heuristic scoring (Zero Face Recognition)
        │
        ▼
[ThreatAssessmentEngine & DeviceThreatDetector]
  ├── Signals: Secondary (+25), Proximity (+20), Persistence (+15),
  │   Privacy zone breach (+15), Approach vector (+15), Primary absence (+25),
  │   Multi-onlooker (+10), Potential screen-capture device detected (+20),
  │   Device inside zone (+15), Device held by secondary (+25)
  ├── Workstation Privacy Zone & Screen-Capture Risk evaluation
  ├── Hysteresis & Debouncing (Smoothing + Deactivation Hold)
  └── Explainable Risk Reasons generation
        │
        ▼
[Active Privacy Defense Layer (ProtectionManager)]
  ├── SAFE / MONITORING: No blocking, subtle telemetry
  ├── WARNING: Non-destructive amber warning privacy shield overlay
  ├── CRITICAL: High-opacity dark screen privacy blackout shield
  ├── CRITICAL + USER ABSENT: Optional native Windows workstation lock
  ├── Deactivation hold & recovery hysteresis
  └── Audit timeline logging
        │
        ▼
[SentinelVisualizer & Modern Desktop Dashboard]
  └── Live AI Acceleration card: Backend, Accelerator (CPU/NPU), Model, Status
```

---

## Qualcomm Snapdragon NPU Deployment Guide

### 1. Strict Architectural Distinction

```text
YOLOv8n PyTorch (.pt)
        ↓
ONNX Export (opset 17, static shape 1x3x640x640)
        ↓
QNN-compatible Conversion / Compilation
        ↓
Qualcomm QNN Context Binary (.bin / .dlc / QNN EP ONNX)
        ↓
ONNX Runtime QNNExecutionProvider (QnnHtp.dll)
        ↓
Verified Snapdragon Hexagon NPU
```

> [!CRITICAL]
> **Exporting to standard ONNX is NOT NPU execution.**
> Ordinary ONNX execution runs on the CPU. Edge Sentinel only reports `Accelerator: NPU` when `QNNExecutionProvider` with the Hexagon Tensor Processor (`QnnHtp.dll`) is verified at runtime. Otherwise, it truthfully reports `Accelerator: CPU` and `Status: FALLBACK`.

### 2. Deployment Status Matrix

| Component | Status | Verification Note |
| :--- | :--- | :--- |
| **Inference Backend Abstraction** | **IMPLEMENTED & VERIFIED** | Validated via `test_phase5_snapdragon_and_fallback.py` |
| **CPU Inference Engine** | **IMPLEMENTED & VERIFIED** | Active on host machine |
| **Snapdragon QNN Engine (`snapdragon_engine.py`)** | **IMPLEMENTED & INTEGRATION-READY** | QNN Execution Provider integration complete |
| **Automatic CPU Fallback** | **IMPLEMENTED & VERIFIED** | Safely triggers on runtime or hardware unavailability |
| **Model Compatibility Validator** | **IMPLEMENTED & VERIFIED** | IoU & confidence delta validation |
| **Dashboard AI Acceleration Card** | **IMPLEMENTED & VERIFIED** | Live hardware status in CustomTkinter sidebar |
| **Snapdragon NPU Benchmark** | **NOT YET VERIFIED** | Requires physical Snapdragon Copilot+ ARM64 PC |

### 3. Steps to Run on Snapdragon Windows Copilot+ PC (e.g., HP OmniBook / EliteBook)

1. **Clone repository onto the Snapdragon ARM64 Windows machine**:
   ```bash
   git clone <repo_url>
   cd HYPERBLOOM
   ```
2. **Install ARM64 dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install onnxruntime-qnn
   ```
3. **Export & Compile Model for Qualcomm Hexagon NPU**:
   ```bash
   python models/qualcomm/export_snapdragon.py
   ```
4. **Configure `config/config.yaml`**:
   ```yaml
   inference:
     backend: "auto"    # or "snapdragon"
   ```
5. **Run Edge Sentinel with live NPU acceleration**:
   ```bash
   python main.py
   ```
6. **Execute benchmark on Snapdragon hardware**:
   ```bash
   python main.py --benchmark --duration 30
   ```

---

## Project Structure

```text
HYPERBLOOM/
├── config/
│   └── config.yaml                     # Centralized config (camera, inference, model, tracking, presence, privacy zone, threat, protection, UI)
├── edge_sentinel/
│   ├── __init__.py
│   ├── config.py                       # Dataclass configuration loader
│   ├── pipeline.py                     # SentinelPipeline coordinator with automatic CPU fallback
│   ├── capture/
│   │   ├── __init__.py
│   │   └── camera.py                   # Thread-safe OpenCV video capture with auto-reconnect
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseInferenceEngine abstract interface
│   │   ├── yolo_engine.py              # Lightweight YOLOv8n engine implementation (CPU)
│   │   ├── snapdragon_engine.py        # Qualcomm Snapdragon QNN engine (Hexagon NPU)
│   │   ├── factory.py                  # Backend selector & CPU fallback manager
│   │   └── validator.py                # Model compatibility validation utility
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py                 # Person and device detection filtering
│   │   └── visualizer.py               # HUD renderer (Privacy zone, bounding boxes, device tags, shield badge)
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py                  # Persistent IoU & Centroid PersonTracker with trajectory history
│   ├── context/
│   │   ├── __init__.py
│   │   ├── schema.py                   # WorkspaceContext & PresenceState schemas
│   │   └── primary_user.py             # PrimaryUserEstimator & presence state machine
│   ├── threat/
│   │   ├── __init__.py
│   │   ├── schema.py                   # ThreatLevel, PrivacyZone, and ThreatAssessment schemas
│   │   └── engine.py                   # ThreatAssessmentEngine with debouncing & explainable reasons
│   │   └── device.py                   # DeviceThreatDetector for smartphone/screen-capture risks
│   ├── protection/
│   │   ├── __init__.py
│   │   ├── schema.py                   # ProtectionAction, ShieldState, ProtectionDecision schemas
│   │   ├── overlay.py                  # Non-destructive Windows privacy shield overlay
│   │   ├── lock.py                     # Native Windows WorkstationLocker with cooldown guardrail
│   │   └── manager.py                  # ProtectionManager coordinating policy & recovery hysteresis
│   └── ui/
│       ├── __init__.py
│       └── dashboard.py                # Modern CustomTkinter dark-mode desktop GUI with AI Acceleration Card
├── models/
│   ├── yolov8n.pt                      # Lightweight YOLOv8 weights (6.2 MB)
│   └── qualcomm/
│       ├── README.md                   # Qualcomm AI Hub / QNN deployment documentation
│       └── export_snapdragon.py        # Static ONNX export & QNN compilation preparation script
├── tests/
│   ├── test_tracking_and_context.py    # Phase 2 unit & integration test suite (7 tests)
│   ├── test_threat_engine.py           # Phase 3 threat assessment test suite (12 tests)
│   ├── test_phase4_protection_and_device.py # Phase 4 protection & device test suite (12 tests)
│   └── test_phase5_snapdragon_and_fallback.py # Phase 5 Snapdragon & fallback test suite (11 tests)
├── main.py                             # Application launcher (GUI & Benchmark modes)
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## How to Run

### 1. Launch the Desktop Application (Phase 5)
```bash
python main.py
```

### 2. Run Hardware Benchmark (Headless 30-Second Execution)
```bash
python main.py --benchmark --duration 30
```

### 3. Run Automated Tests (42 Total Tests)
```bash
python -m pytest -v
```

---

## Performance Comparison (Host Development Machine: AMD64 Windows 11)

Measured on physical webcam capture (30-second duration):

| Metric | Phase 4 (CPU Baseline) | Phase 5 (Auto Mode on AMD64 Host) | Phase 5 (Target Snapdragon NPU) |
| :--- | :--- | :--- | :--- |
| **Backend** | YOLOv8n (CPU) | **YOLOv8n (CPU Fallback)** | **Snapdragon QNN (HTP)** |
| **Accelerator** | Host CPU | **CPU (Fallback Active)** | **Qualcomm Hexagon NPU** |
| **Resolution** | 640 × 480 @ 30 FPS target | 640 × 480 @ 30 FPS target | 640 × 480 @ 30 FPS target |
| **Frames Processed** | 280 frames (30s) | **415 frames (30s)** | Target machine required |
| **Throughput (FPS)** | 9.34 FPS | **13.82 FPS** | Target machine required |
| **Inference Latency** | 106.57 ms | **72.00 ms** (Min: 56.0 ms, Max: 153.1 ms) | Target machine required |
| **Tracking Overhead** | 0.06 ms | **0.01 ms** | Target machine required |
| **Threat Overhead** | 0.06 ms | **0.04 ms** | Target machine required |
| **Protection Overhead**| 0.03 ms | **0.02 ms** | Target machine required |
| **Total Latency** | 106.72 ms | **72.08 ms** | Target machine required |
| **CPU Usage** | ~65% | **60.2%** (392.9 MB RAM) | Target machine required |
| **Cloud Transmission** | 0.00% (Local) | **0.00% (Strictly Local)** | 0.00% (Strictly Local) |
| **Verification Status**| Measured | **Measured & Verified** | **NOT EXECUTED (No ARM64 hardware)** |

