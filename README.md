# Edge Sentinel — Privacy-Preserving On-Device AI Security Agent
### Hardware-Validated on Qualcomm Snapdragon X2 Elite (Hexagon v81 NPU)

[![Snapdragon X2 Elite](https://img.shields.io/badge/Hardware-Snapdragon%20X2%20Elite-blue?style=for-the-badge&logo=qualcomm)](https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-x-elite)
[![Hexagon NPU](https://img.shields.io/badge/Accelerator-Hexagon%20v81%20NPU-purple?style=for-the-badge)](https://www.qualcomm.com/products/technology/processors/hexagon-processor)
[![QNN Provider](https://img.shields.io/badge/Provider-QNNExecutionProvider-green?style=for-the-badge)](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
[![Inference Latency](https://img.shields.io/badge/YOLOv8n%20Inference-3.23%20ms-brightgreen?style=for-the-badge)](#real-hardware-performance-benchmark)
[![Pipeline Throughput](https://img.shields.io/badge/Pipeline%20Throughput-146.84%20FPS-orange?style=for-the-badge)](#real-hardware-performance-benchmark)
[![Tests](https://img.shields.io/badge/Tests-42%2F42%20Passed-success?style=for-the-badge)](#test-suite--regression-validation)
[![Privacy Validation](https://img.shields.io/badge/Privacy-0%20Network%20Calls%20Observed-blue?style=for-the-badge)](#privacy-by-design--security-architecture)

---

## Executive Summary

**Edge Sentinel** is an autonomous, privacy-first on-device AI security guardian engineered for enterprise workstations, financial terminals, and privacy-sensitive mobile professionals. By leveraging the **Qualcomm Hexagon v81 NPU** on the **Snapdragon X2 Elite (X2E88100)** Copilot+ PC architecture, Edge Sentinel continuously monitors for **shoulder surfing**, **physical privacy zone breaches**, and **potential screen-capture risks** with ultra-low millisecond latency and near-zero CPU overhead.

### Key Highlights
- **Real-Hardware Validated**: Verified on a physical Qualcomm Compute Reference Design SC8480XP / MTP running Windows 11 Enterprise ARM64.
- **Millisecond NPU Acceleration**: **3.23 ms** standalone YOLOv8n inference (**309.23 FPS** processing throughput) and **5.55 ms** complete end-to-end pipeline latency (**146.84 FPS** processing throughput) running on the Hexagon NPU (with **3.03–3.46 ms** standalone latency and **146.84–193.67 FPS** pipeline throughput observed across multiple verified benchmark runs).
- **Zero CPU Bottleneck**: Only **7.0% – 10.8% CPU utilization** during continuous full-pipeline AI vision, tracking, context estimation, threat scoring, and defense.
- **Strict Privacy Architecture**: Camera frames are processed strictly in volatile memory and immediately discarded. 0 network calls observed during the real-hardware validation benchmark, zero raw frame disk persistence, zero facial recognition, and zero biometric embeddings.
- **Native ARM64 Optimization**: The Snapdragon inference pipeline runs natively on Windows 11 ARM64 using Pillow for letterbox preprocessing and pure NumPy vectorized NMS, requiring zero OpenCV dependencies. (OpenCV is reserved solely for optional webcam video capture).
- **Transparent Fallback Architecture**: Pluggable inference layer auto-probes Qualcomm QNN hardware and seamlessly falls back to CPU if unavailable, guaranteeing 100% stability across platforms.

---

## System Architecture Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EDGE SENTINEL COMPLETE PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                         [Thread-Safe Frame Capture]
                         (Webcam DirectShow / RAM)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PLUGGABLE INFERENCE BACKEND                           │
│  ┌────────────────────────────────┐  ┌───────────────────────────────────┐  │
│  │  Qualcomm Snapdragon Backend   │  │            CPU Backend            │  │
│  │  Hexagon v81 NPU / QnnHtp.dll  │  │ ONNX Runtime CPUExecutionProvider │  │
│  │  (3.23 ms YOLOv8n Inference)   │  │        (Safe Auto-Switch)         │  │
│  └────────────────────────────────┘  └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                         [Person & Device Detection]
                         (COCO Class 0: Person)
                         (COCO Class 67: Cell Phone)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRACKING & CONTEXT ESTIMATION                         │
│  ├── Centroid & IoU Association (Persistent Track IDs)                      │
│  ├── Workspace Context State Machine (ACTIVE ⇆ TEMP_ABSENT ⇆ ABSENT)       │
│  └── Primary User Estimation (Spatial / Temporal Heuristic Scoring)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THREAT ASSESSMENT ENGINE                            │
│  ├── Multi-Signal Threat Evaluation (Proximity, Persistence, Approach)     │
│  ├── Workstation Privacy Zone Breach Analysis                               │
│  ├── Potential Screen-Capture Risk Detection                                │
│  ├── Temporal Debounce, Exponential Smoothing, and Hold State Machine       │
│  └── Explainable Risk Reasoning Generator                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ACTIVE DEFENSE & PROTECTION                         │
│  ├── SAFE (Score < 30)       → Normal Status, Clean Display                 │
│  ├── MONITORING (Score 30-59)→ Awareness Logging, HUD Indicator             │
│  ├── WARNING (Score 60-79)   → Non-Destructive Amber Warning Privacy Shield │
│  ├── CRITICAL (Score 80-100) → High-Opacity Blackout Privacy Shield         │
│  └── CRITICAL + USER ABSENT  → Optional Native Windows Workstation Lock     │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                          [Modern Desktop Dashboard]
                          (CustomTkinter HUD Interface)
```

---

## Threat Scoring Engine & Heuristic Weights

Edge Sentinel implements a multi-signal heuristic threat assessment engine that evaluates physical surroundings in real time without invasive biometric identification.

### 1. Threat Scoring Formula

$$\text{ThreatScore} = \min\left(100, \sum w_i \cdot S_i \times \text{Multiplier}_{\text{absence}}\right)$$

Where $w_i$ represents the individual signal weight, $S_i \in [0, 1]$ represents the normalized signal intensity, and $\text{Multiplier}_{\text{absence}}$ dynamically elevates threat severity when the primary user leaves the workstation.

### 2. Multi-Signal Threat Matrix

| Signal ID | Threat Factor | Trigger Condition | Weight Contribution | Rationale |
| :--- | :--- | :--- | :---: | :--- |
| `SIG_SECONDARY` | Secondary Person Present | Any secondary person detected in view | **+25** | Baseline shoulder-surfing risk |
| `SIG_PROXIMITY` | Proximity Risk | Secondary person distance to workstation center (< 0.45) | **0 to +20** | Closer individuals pose higher visual risk |
| `SIG_PERSISTENCE` | Persistence Risk | Secondary person present for $\ge 4.0$ seconds | **+15** | Filters transient background passersby |
| `SIG_ZONE` | Privacy Zone Breach | Secondary person enters bounding box of Privacy Zone | **+15** | Direct physical intrusion into screen field |
| `SIG_APPROACH` | Approach Vector | Secondary person moving toward privacy zone ($\ge 15.0$ px/s) | **+15** | Active intent to approach display |
| `SIG_ABSENT` | Primary User Absence | Primary user leaves workstation while secondary present | **+25** | Unattended sensitive workspace vulnerability |
| `SIG_MULTI` | Multi-Onlooker Risk | $\ge 2$ secondary persons simultaneously in view | **+10** | Group visual capture hazard |
| `SIG_DEVICE` | Potential Screen-Capture | Cell phone detected in scene (COCO Class 67) | **+20** | Handheld optical recording hazard |
| `SIG_DEV_ZONE` | Device Inside Privacy Zone | Cell phone located within workstation Privacy Zone | **+15** | High-probability screen recording angle |
| `SIG_DEV_HELD` | Device Held by Secondary | Secondary person IoU overlaps detected phone | **+25** | Active physical photography posture |
| `SIG_DEV_ABSENT`| Device While User Absent | Cell phone detected while primary user is absent | **+20** | Unauthorized optical capture while desk unattended |

### 3. Threat Levels & Defense Mapping

The discrete threat levels map directly to configuration thresholds (`threshold_monitoring: 30.0`, `threshold_warning: 60.0`, `threshold_critical: 80.0`):

| Threat Level | Score Range | Shield State | Protection Action | Visualizer Display |
| :--- | :---: | :---: | :--- | :--- |
| **SAFE** | $0.0 - 29.9$ | Inactive | `NONE` | Green boundary, clean HUD display |
| **MONITORING** | $30.0 - 59.9$ | Inactive | `MONITORING_NOTICE` | Yellow indicator, awareness logging |
| **WARNING** | $60.0 - 79.9$ | **Active** | `WARNING_OVERLAY` | Amber warning shield, explainable alerts |
| **CRITICAL** | $80.0 - 100.0$ | **Active** | `STRONG_OVERLAY` | High-opacity blackout shield, deterrent |
| **LOCKDOWN** | $80.0 - 100.0$ (User Absent)| **Active** | `WORKSTATION_LOCK` | Optional Native Windows Lock (`LockWorkStation`) |

### 4. Debouncing, Smoothing & Recovery Hysteresis
- **Score Smoothing**: Instantaneous threat scores are smoothed via an exponential moving average ($S_t = 0.40 \cdot S_{\text{raw}} + 0.60 \cdot S_{t-1}$) to suppress frame-to-frame noise.
- **Deactivation Hold Time**: Once elevated, the threat level remains held for a minimum of $2.0$ seconds (`deactivation_hold_sec: 2.0`) before stepping down to prevent jitter when an onlooker turns away briefly.
- **Protection Shield Recovery Hold**: The privacy shield overlay enforces a $3.0$-second recovery grace period (`recovery_hold_sec: 3.0`) when stepping down towards a lower state to prevent visual screen flickering.
- **Workstation Lock Guardrails**: Native workstation locking is disabled by default (`lock_workstation_enabled: false`) as a safety guardrail. When enabled, it triggers only if the primary user is absent during a critical threat, with a 60-second cooldown (`lock_cooldown_sec: 60.0`).

---

## Real Hardware Performance Benchmark

### Verification Platform
- **Device**: Qualcomm Compute Reference Design SC8480XP / MTP
- **SoC**: Qualcomm Snapdragon X2 Elite X2E88100
- **NPU**: Qualcomm Hexagon v81 NPU (HTP Architecture)
- **CPU**: 18 Oryon CPU Cores (ARM64)
- **OS**: Windows 11 Enterprise (Build 28000)
- **Runtime**: Python 3.12.9 ARM64 + `onnxruntime-qnn==1.24.4` (`cp312-win_arm64`)
- **Backend**: `QNNExecutionProvider` with `QnnHtp.dll`

### Benchmark Results (Real Physical Hardware)

| Metric | Standalone YOLOv8n NPU Inference | Complete Edge Sentinel Security Pipeline | Explicit CPU Execution (`CPUExecutionProvider`) |
| :--- | :---: | :---: | :---: |
| **Execution Accelerator** | **Qualcomm Hexagon v81 NPU** | **Qualcomm Hexagon v81 NPU** | **Oryon CPU (ARM64)** |
| **Runtime Provider** | `QNNExecutionProvider` (`QnnHtp.dll`) | `QNNExecutionProvider` (`QnnHtp.dll`) | `CPUExecutionProvider` |
| **CPU Fallback** | **NO** (Pure NPU Execution) | **NO** (Pure NPU Execution) | Active CPU baseline |
| **Input Shape** | Static `1x3x640x640` (float32) | Static `1x3x640x640` (float32) | Static `1x3x640x640` (float32) |
| **Average Latency** | **3.23 ms** | **5.55 ms** | **52.57 ms** (Inference) / **52.61 ms** (Pipeline) |
| **Observed Latency Range**| **3.03 ms – 3.46 ms** | **5.14 ms – 5.55 ms** | **38.49 ms – 111.67 ms** |
| **Processing Throughput** | **309.23 FPS** (up to 330.07 FPS) | **146.84 FPS** (up to 193.67 FPS) | **18.99 FPS** |
| **Tracking Overhead** | — | **0.00 ms** (Sub-millisecond) | **0.01 ms** |
| **Threat Assessment Overhead**| — | **0.01 ms** (Sub-millisecond) | **0.02 ms** |
| **Protection Manager Overhead**| — | **0.01 ms** (Sub-millisecond) | **0.01 ms** |
| **CPU Utilization** | < 4% | **7.0% – 10.8%** | **26.9%** (Single-core compute bound) |
| **Process Memory (RSS)** | 215.0 MB | **463.8 MB – 481.2 MB** | **195.9 MB** |
| **Network Observations** | **0 Network Calls Observed** | **0 Network Calls Observed** | **0 Network Calls Observed** |

> [!NOTE]
> **Processing Throughput vs. Physical Camera Capture FPS:**
> The **146.84 FPS** and **193.67 FPS** figures represent benchmark **processing throughput** measured during full pipeline evaluation. They demonstrate the compute capacity of the Hexagon NPU. Physical webcam hardware captures video at its standard configured rate (~30 FPS).

---

## Real Threat Scenario Validation on NPU

All security scenarios were evaluated on the real Snapdragon X2 Elite Hexagon NPU:

| Scenario | Conditions | Threat Score | Threat Level | Privacy Shield | Protection Action | Recovery Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **A. Primary User Only** | Primary user alone in workspace | **0.0** | `SAFE` | False | `NONE` | Baseline safe state |
| **B. Secondary Enters** | Secondary person enters room background | **25.0** | `SAFE` | False | `NONE` | Baseline monitoring point |
| **C. Secondary Approaches** | Secondary walks toward desk ($>4\text{s}$, close) | **60.0** | `WARNING` | **True** | `WARNING_OVERLAY` | Amber warning shield activates |
| **D. Privacy Zone Breach** | Secondary enters workstation privacy zone | **75.0** | `WARNING` | **True** | `WARNING_OVERLAY` | Zone breach alert generated |
| **E. Screen-Capture Risk**| Phone detected near active screen | **100.0** | `CRITICAL` | **True** | `STRONG_OVERLAY` | Blackout shield deployed |
| **F. Primary Absent + Secondary**| Primary leaves while secondary present | **100.0** | `CRITICAL` | **True** | `STRONG_OVERLAY` | Eligible for workstation lock |
| **G. Secondary Leaves** | Secondary leaves field of view | **0.0** | `SAFE` | False | `NONE` | Smooth recovery after hysteresis |

> [!IMPORTANT]
> **Technical Threat Model Terminology:**
> - Edge Sentinel assesses **potential physical privacy risks**, **shoulder-surfing risks**, **proximity risks**, and **privacy-zone breaches**.
> - Detection of a smartphone indicates a **potential screen-capture risk**; the system identifies COCO class 67 (cell phone) and does **not** claim or infer that the device is actively recording.
> - The application distinguishes **primary user** vs. **secondary persons** via spatial and temporal proximity heuristics without facial recognition or biometric enrollment.

---

## Privacy by Design & Security Architecture

Designed for **on-device privacy**:

```text
Camera Frame
    ↓ (volatile in-memory buffer only — zero disk write)
Local Preprocessing (Pillow / NumPy)
    ↓
Local AI Inference (Qualcomm Hexagon NPU via QNN)
    ↓
Detection, Tracking & Workspace Context Engine
    ↓
Threat Assessment & Explainable Risk Scoring
    ↓
Local Privacy Defense (Visualizer / Non-destructive Privacy Shield)
    ↓
Local Metadata & Telemetry (No raw frames retained)
```

- **0 Network Calls**: 0 network calls were observed during the real-hardware validation benchmark.
- **Zero Raw Frame Writes**: Camera frames are processed strictly in RAM and never written to disk or persistent storage.
- **Zero Facial Recognition**: No biometric embeddings, face templates, or identity profiles are generated or stored.
- **Strictly Local Telemetry**: Only bounding-box metadata, risk scores, and accelerator runtime metrics are logged.

---

## Camera Capture vs. Snapdragon Inference Architecture

To ensure high performance and cross-platform reliability, Edge Sentinel cleanly separates video capture from AI inference:

1. **Camera Ingestion Layer (`edge_sentinel/capture/camera.py`)**:
   - Uses OpenCV (`cv2.VideoCapture` with DirectShow backend) when running on a physical system with an attached webcam.
   - Operates on a dedicated, thread-safe background capture loop.
   - If OpenCV is absent or no camera is attached (such as in headless CI/CD environments or cloud VMs), the pipeline gracefully generates in-memory synthetic frames to allow hardware benchmarking without crashes.
2. **Qualcomm Snapdragon Inference Engine (`edge_sentinel/inference/snapdragon_engine.py`)**:
   - Operates with **zero OpenCV dependencies**.
   - Frame letterboxing and resizing are performed natively via **Pillow** (`PIL.Image.Resampling.BILINEAR`).
   - Normalization, transposition, and Non-Maximum Suppression (NMS) are implemented using pure, vectorized **NumPy**.
   - Directly dispatches static `1x3x640x640` float32 tensors to ONNX Runtime's `QNNExecutionProvider` targeting the Hexagon Tensor Processor via `QnnHtp.dll`.

---

## Repository Structure

```text
edge-sentinel/
├── .gitignore                          # Protects keys, credentials, local envs, and *.onnx artifacts
├── LICENSE                             # MIT Open Source License
├── README.md                           # Master architectural & benchmark documentation
├── START_EDGE_SENTINEL.bat             # One-click Windows launcher with priority venv detection
├── requirements.txt                    # Project dependencies
├── pytest.ini                          # Automated test discovery configuration
├── main.py                             # Application entrypoint (GUI Dashboard & Headless Benchmark)
│
├── config/
│   └── config.yaml                     # Centralized configuration (Camera, Model, Threat, Protection, UI)
│
├── edge_sentinel/                      # Core Edge Sentinel Engine
│   ├── __init__.py
│   ├── config.py                       # Dataclass configuration loader
│   ├── pipeline.py                     # Master SentinelPipeline coordinator with CPU fallback
│   │
│   ├── capture/                        # Video Ingestion Layer
│   │   ├── __init__.py
│   │   └── camera.py                   # Thread-safe DirectShow OpenCV capture with auto-reconnect
│   │
│   ├── inference/                      # Pluggable AI Acceleration Layer
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseInferenceEngine abstract interface
│   │   ├── factory.py                  # Backend selector & automatic CPU fallback manager
│   │   ├── snapdragon_engine.py        # Qualcomm QNN Hexagon NPU engine (Pillow + pure NumPy NMS)
│   │   ├── validator.py                # Model compatibility & schema validation utility
│   │   └── yolo_engine.py              # CPU inference engine (PyTorch / ONNX Runtime CPU fallback)
│   │
│   ├── detection/                      # Computer Vision Layer
│   │   ├── __init__.py
│   │   ├── detector.py                 # Person (class 0) & Cell phone (class 67) filter
│   │   └── visualizer.py               # HUD visualizer (Privacy Zone, bounding boxes, shield badges)
│   │
│   ├── tracking/                       # Object Tracking Layer
│   │   ├── __init__.py
│   │   └── tracker.py                  # Persistent IoU & Centroid tracker with trajectory history
│   │
│   ├── context/                        # Workspace Context Layer
│   │   ├── __init__.py
│   │   ├── primary_user.py             # PrimaryUserEstimator & presence state machine
│   │   └── schema.py                   # PresenceState & WorkspaceContext data schemas
│   │
│   ├── threat/                         # Threat Assessment Layer
│   │   ├── __init__.py
│   │   ├── device.py                   # DeviceThreatDetector for screen-capture device risks
│   │   ├── engine.py                   # ThreatAssessmentEngine with debouncing & explainable reasons
│   │   └── schema.py                   # ThreatLevel, PrivacyZone, and ThreatAssessment schemas
│   │
│   ├── protection/                     # Active Privacy Defense Layer
│   │   ├── __init__.py
│   │   ├── lock.py                     # Native Windows WorkstationLocker guardrail
│   │   ├── manager.py                  # ProtectionManager coordinating hysteresis & recovery
│   │   ├── overlay.py                  # Non-destructive Windows privacy shield overlay
│   │   └── schema.py                   # ProtectionAction & ShieldState schemas
│   │
│   └── ui/                             # Presentation Layer
│       ├── __init__.py
│       └── dashboard.py                # Modern CustomTkinter dark-mode desktop GUI
│
├── models/                             # Model Storage & Compilation Tooling
│   ├── yolov8n.pt                      # PyTorch YOLOv8n base weights (Tracked in Git, 6.2 MB)
│   └── qualcomm/
│       ├── README.md                   # Qualcomm Snapdragon compilation & deployment guide
│       ├── export_snapdragon.py        # Static ONNX graph export utility (1x3x640x640, opset 17)
│       └── yolov8n.onnx                # Generated ONNX artifact (Created via export_snapdragon.py; gitignored)
│
└── tests/                              # Comprehensive Test Suite (42 Tests)
    ├── test_phase4_protection_and_device.py   # Device threat & privacy shield tests (12 tests)
    ├── test_phase5_snapdragon_and_fallback.py # Snapdragon QNN engine & fallback tests (11 tests)
    ├── test_threat_engine.py                  # Threat scoring, weights & debounce tests (12 tests)
    └── test_tracking_and_context.py           # Centroid tracking & primary presence tests (7 tests)
```

---

## Test Suite & Regression Validation

Edge Sentinel includes 42 automated tests covering tracking, presence state transitions, threat weight scoring, device detection, privacy shield hysteresis, and Snapdragon QNN fallback mechanics:

```powershell
python -m pytest -q
..........................................                               [100%]
42 passed in 8.56s
```

### Coverage Breakdown
- **Tracking & Context** (`test_tracking_and_context.py`): 7 / 7 passed
- **Threat Engine** (`test_threat_engine.py`): 12 / 12 passed
- **Protection & Device Threat** (`test_phase4_protection_and_device.py`): 12 / 12 passed
- **Snapdragon & Fallback** (`test_phase5_snapdragon_and_fallback.py`): 11 / 11 passed
- **Total**: **42 / 42 tests passed** (100%)

---

## Quickstart & Reproducibility Guide

### 1. Target Hardware: Qualcomm Snapdragon ARM64 Windows 11 Copilot+ PC

#### Step A: Verify Environment
```powershell
python --version
# Expected: Python 3.12.x (ARM64)

python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
# Expected: 1.24.4, ['QNNExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

#### Step B: Install ARM64 Dependencies
```powershell
pip install -r requirements.txt
pip install onnxruntime-qnn==1.24.4
```
*(Note: `onnxruntime-qnn==1.24.4` installs the native Windows ARM64 wheel with Hexagon HTP support).*

#### Step C: Generate Static ONNX Model for QNN
The source weights `models/yolov8n.pt` are tracked in Git. Run the export script to generate the static-shape ONNX model required by QNN:
```powershell
python models/qualcomm/export_snapdragon.py --weights models/yolov8n.pt --output_dir models/qualcomm
```

#### Step D: Run Full Automated Test Suite
```powershell
pytest -q
```

#### Step E: Execute Hardware Benchmark
```cmd
START_EDGE_SENTINEL.bat --benchmark --duration 30
:: Or: python main.py --benchmark --duration 30
```

#### Step F: Launch Edge Sentinel Desktop Dashboard
```cmd
START_EDGE_SENTINEL.bat
:: Or: python main.py
```

---

### 2. Standard Development PC (x86_64 CPU Fallback)

Edge Sentinel supports standard x86_64 development machines (Windows, macOS, Linux). When Snapdragon NPU hardware is not detected, the system activates automatic CPU fallback without configuration changes:

```cmd
# 1. Install standard dependencies
pip install -r requirements.txt

# 2. Run test suite
pytest -q

# 3. Launch with one-click launcher or python
START_EDGE_SENTINEL.bat
:: Or: python main.py
```

---

## Configuration Reference (`config/config.yaml`)

Below is the complete configuration schema as defined in `config/config.yaml`:

```yaml
# Edge Sentinel Configuration
# Designed for local on-device privacy security agent

camera:
  device_index: 0              # Physical webcam index (DirectShow)
  width: 640                   # Capture width in pixels
  height: 480                  # Capture height in pixels
  fps: 30                      # Camera target framerate
  auto_reconnect: true         # Automatically attempt reconnection if stream drops
  reconnect_delay_sec: 2.0     # Delay between reconnection attempts in seconds

inference:
  backend: "auto"              # Inference backend: 'auto', 'snapdragon', or 'cpu'
  qnn_backend_path: "QnnHtp.dll" # Qualcomm HTP backend library name
  allow_fallback: true         # Automatically fall back to CPU if Snapdragon is unavailable

model:
  engine: "yolov8"             # Detector family
  model_path: "models/yolov8n.pt" # Local PyTorch weights file
  qnn_model_path: "models/qualcomm/yolov8n.onnx" # Static ONNX model path for QNN
  backend: "auto"              # Model-specific backend override
  device: "cpu"                # Target device for CPU engine
  input_size: 640              # Model input dimension (640x640)
  target_classes: [0, 67]      # COCO class IDs: 0 = person, 67 = cell phone

detection:
  confidence_threshold: 0.50   # Minimum detection confidence
  iou_threshold: 0.45          # NMS IoU overlap threshold
  max_detections: 10           # Maximum detections to retain per frame

tracking:
  max_missing_frames: 20       # Frames before a lost track is purged (~0.6-1.0 sec)
  iou_match_threshold: 0.30    # Minimum IoU for track association
  dist_match_threshold: 120.0  # Pixel distance threshold for centroid matching

presence:
  temp_absence_threshold_sec: 2.5 # Time before primary state transitions: ACTIVE -> TEMPORARILY_ABSENT
  absent_threshold_sec: 7.0       # Time before primary state transitions: TEMPORARILY_ABSENT -> ABSENT
  center_weight: 0.40            # Preference weight for person centered in webcam field of view
  proximity_weight: 0.35         # Preference weight for person close to laptop (larger bounding box)
  persistence_weight: 0.25       # Preference weight for person visible continuously

privacy_zone:
  enabled: true                # Enable workstation privacy boundary
  x_min: 0.25                  # Normalized left boundary (25% from left)
  y_min: 0.15                  # Normalized top boundary (15% from top)
  x_max: 0.75                  # Normalized right boundary (75% from left)
  y_max: 0.90                  # Normalized bottom boundary (90% from top)

threat:
  weight_secondary_detected: 25.0       # Baseline points when secondary person is present
  weight_close_workstation: 20.0        # Points when secondary is in close proximity (< 0.45)
  weight_persistent_presence: 15.0      # Points when secondary lingers >= 4.0s
  weight_inside_privacy_zone: 15.0      # Points when secondary breaches privacy zone
  weight_approaching_privacy_zone: 15.0 # Points when secondary moves toward zone >= 15 px/s
  weight_primary_absent: 25.0           # Points when primary is absent while secondary is near
  weight_multiple_secondary: 10.0       # Points when 2 or more secondary persons are in view
  proximity_alert_threshold: 0.45       # Normalized distance threshold for proximity alert
  persistence_duration_threshold_sec: 4.0 # Time threshold in seconds for persistent onlooker
  approach_velocity_threshold: 15.0     # Approach velocity threshold (pixels/second)
  smoothing_factor: 0.40                # Exponential moving average score smoothing factor
  deactivation_hold_sec: 2.0            # Minimum hold duration for elevated threat level
  threshold_monitoring: 30.0            # Score threshold to enter MONITORING (30.0 - 59.9)
  threshold_warning: 60.0               # Score threshold to enter WARNING (60.0 - 79.9)
  threshold_critical: 80.0              # Score threshold to enter CRITICAL (80.0 - 100.0)
  device:
    enabled: true                       # Enable potential screen-capture device detection
    weight_device_detected: 20.0        # Points when cell phone is detected
    weight_device_inside_zone: 15.0     # Points when cell phone enters privacy zone
    weight_device_held_by_secondary: 25.0 # Points when cell phone is held by secondary person
    weight_device_primary_absent: 20.0  # Points when cell phone is detected while primary is absent
    proximity_threshold: 0.015          # IoU overlap threshold between person and device

protection:
  enabled: true                         # Enable active privacy defense system
  warning_overlay_enabled: true         # Deploy amber privacy shield on WARNING
  strong_overlay_enabled: true          # Deploy blackout privacy shield on CRITICAL
  lock_workstation_enabled: false       # Safety guardrail: native lock disabled by default
  lock_require_user_absent: true        # Guardrail: lock only if user is confirmed absent
  lock_cooldown_sec: 60.0               # Cooldown between lock triggers in seconds
  recovery_hold_sec: 3.0                # Shield hold duration in seconds during recovery
  warning_alpha: 0.85                   # Opacity for amber warning shield overlay
  strong_alpha: 0.95                   # Opacity for high-opacity blackout shield overlay

ui:
  theme: "dark"                         # Dashboard color theme
  window_title: "Edge Sentinel — Privacy-Preserving On-Device AI Security Agent"
  window_width: 1220                    # GUI window width in pixels
  window_height: 760                    # GUI window height in pixels
  display_fps: true                     # Show live FPS counter on video feed
  display_latency: true                 # Show inference latency on video feed
  display_privacy_badge: true           # Display on-device privacy guarantee badge
```

---

## License & Third-Party Attributions

- **Edge Sentinel**: Distributed under the [MIT License](LICENSE).
- **YOLOv8**: Developed by [Ultralytics](https://github.com/ultralytics/ultralytics) under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html). Commercial deployments requiring proprietary terms must secure a commercial license directly from Ultralytics.
- **Qualcomm Snapdragon & Hexagon**: Registered trademarks of Qualcomm Incorporated. QNN SDK and Hexagon libraries are proprietary to Qualcomm Technologies, Inc.
