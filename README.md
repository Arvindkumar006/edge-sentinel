# Edge Sentinel — Privacy-Preserving On-Device AI Security Agent
### Real Hardware-Validated on Qualcomm Snapdragon X2 Elite (Hexagon v81 NPU)

[![Snapdragon X2 Elite](https://img.shields.io/badge/Hardware-Snapdragon%20X2%20Elite-blue?style=for-the-badge&logo=qualcomm)](https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-x-elite)
[![Hexagon NPU](https://img.shields.io/badge/Accelerator-Hexagon%20v81%20NPU-purple?style=for-the-badge)](https://www.qualcomm.com/products/technology/processors/hexagon-processor)
[![QNN Provider](https://img.shields.io/badge/Provider-QNNExecutionProvider-green?style=for-the-badge)](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
[![Inference Latency](https://img.shields.io/badge/YOLOv8n%20Inference-3.23%20ms-brightgreen?style=for-the-badge)](#performance-benchmark-results)
[![Pipeline Throughput](https://img.shields.io/badge/Pipeline%20Throughput-146.84%20FPS-orange?style=for-the-badge)](#performance-benchmark-results)
[![Tests](https://img.shields.io/badge/Tests-42%2F42%20Passed-success?style=for-the-badge)](#test-suite--regression-validation)
[![Zero Cloud](https://img.shields.io/badge/Privacy-100%25%20On--Device-red?style=for-the-badge)](#privacy-by-design--security-architecture)

---

## Executive Summary

**Edge Sentinel** is an autonomous, privacy-first on-device AI security guardian engineered for enterprise workstations, financial terminals, and privacy-sensitive mobile professionals. By leveraging the **Qualcomm Hexagon v81 NPU** on the **Snapdragon X2 Elite (X2E88100)** Copilot+ PC architecture, Edge Sentinel continuously monitors for **shoulder surfing**, **physical privacy zone breaches**, and **potential screen-capture risks** with ultra-low millisecond latency and near-zero CPU overhead.

### Key Highlights
- **Real-Hardware Validated**: 100% verified on physical Qualcomm Compute Reference Design SC8480XP / MTP running Windows 11 Enterprise ARM64.
- **Millisecond NPU Acceleration**: **3.03 ms - 3.23 ms** standalone YOLOv8n inference (**309.23 - 330.07 FPS** processing throughput) and **5.12 ms - 5.55 ms** complete end-to-end pipeline latency (**146.84 - 193.94 FPS** processing throughput) running on the Hexagon NPU.
- **Zero CPU Bottleneck**: Only **10.8% CPU utilization** during continuous full-pipeline AI vision, tracking, context estimation, threat scoring, and defense.
- **Zero-Cloud Privacy Guarantee**: Camera frames are processed strictly in-memory and immediately discarded. Zero network calls, zero raw frame disk persistence, zero facial recognition, and zero biometric embeddings.
- **Native ARM64 Optimization**: Inference path uses Pillow + pure NumPy vectorized NMS without OpenCV dependencies, ensuring seamless execution on Windows 11 ARM64.
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
│  ┌─────────────────────────────────────┐   ┌─────────────────────────────┐  │
│  │     Qualcomm Snapdragon Backend     │   │         CPU Backend         │  │
│  │   Hexagon v81 NPU / QnnHtp.dll      │   │   PyTorch / ONNX Fallback   │  │
│  │     (3.23 ms YOLOv8n Inference)     │   │      (Safe Auto-Switch)     │  │
│  └─────────────────────────────────────┘   └─────────────────────────────┘  │
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
│  ├── Temporal Debounce, Hysteresis, and Recovery State Machine              │
│  └── Explainable Risk Reasoning Generator                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ACTIVE DEFENSE & PROTECTION                         │
│  ├── SAFE (Score < 40)      → Telemetry Logging, HUD Indicator              │
│  ├── WARNING (Score 40-79)  → Non-Destructive Amber Warning Privacy Shield  │
│  ├── CRITICAL (Score 80-100)→ High-Opacity Blackout Privacy Shield          │
│  └── CRITICAL + ABSENT      → Optional Native Windows Workstation Lock      │
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
| `SIG_PROXIMITY` | Proximity Risk | Secondary person distance to workstation center | **0 to +20** | Closer individuals pose higher visual risk |
| `SIG_PERSISTENCE` | Persistence Risk | Secondary person present for $> 3.0$ seconds | **+15** | Filters transient background passersby |
| `SIG_ZONE` | Privacy Zone Breach | Secondary person enters bounding box of Privacy Zone | **+15** | Direct physical intrusion into screen field |
| `SIG_APPROACH` | Approach Vector | Secondary person bounding box expanding towards camera | **+15** | Active intent to approach display |
| `SIG_ABSENT` | Primary User Absence | Primary user leaves workstation while secondary present | **+25** | Unattended sensitive workspace vulnerability |
| `SIG_MULTI` | Multi-Onlooker Risk | $\ge 2$ secondary persons simultaneously in view | **+10** | Group visual capture hazard |
| `SIG_DEVICE` | Potential Screen-Capture | Cell phone detected in scene (COCO Class 67) | **+20** | Handheld optical recording hazard |
| `SIG_DEV_ZONE` | Device Inside Privacy Zone | Cell phone located within workstation Privacy Zone | **+15** | High-probability screen recording angle |
| `SIG_DEV_HELD` | Device Held by Secondary | Secondary person IoU overlaps detected phone | **+25** | Active physical photography posture |

### 3. Threat Levels & Defense Mapping

| Threat Level | Score Range | Shield State | Protection Action | Visualizer Display |
| :--- | :---: | :---: | :--- | :--- |
| **SAFE** | $0 - 39$ | Inactive | `NONE` | Green boundary, live FPS/latency |
| **WARNING** | $40 - 79$ | **Active** | `WARNING_OVERLAY` | Amber warning shield, explainable alerts |
| **CRITICAL** | $80 - 100$ | **Active** | `STRONG_OVERLAY` | High-opacity blackout shield, audio deterrent |
| **LOCKDOWN** | $100$ (User Away)| **Active** | `LOCK_WORKSTATION` | Native Windows Lock Workstation (`LockWorkStation`) |

### 4. Debouncing, Hysteresis & Recovery
- **Activation Debounce**: Transient visual spikes are filtered using a temporal smoothing window ($N=5$ frames).
- **Deactivation Hold Time**: Once activated, the privacy shield remains locked for a minimum configurable cooldown ($2.0$ seconds default) to prevent screen flickering while an onlooker departs.
- **Recovery Hysteresis**: Score must drop and sustain below the safe threshold ($30.0$) before shield deactivation occurs.

---

## Real Hardware Performance Benchmark

### Verification Platform
- **Device**: Qualcomm Compute Reference Design SC8480XP / MTP
- **SoC**: Qualcomm Snapdragon X2 Elite X2E88100
- **NPU**: Qualcomm Hexagon v81 NPU (HTP Architecture)
- **CPU**: 18 Oryon CPU Cores (ARM64)
- **OS**: Windows 11 Enterprise (Build 28000.1764)
- **Runtime**: Python 3.12.9 ARM64 + `onnxruntime-qnn==1.24.4` (`cp312-win_arm64`)
- **Backend**: `QNNExecutionProvider` with `QnnHtp.dll`

### Benchmark Results (Real Physical Hardware)

| Metric | Standalone YOLOv8n NPU Inference | Complete Edge Sentinel Security Pipeline |
| :--- | :---: | :---: |
| **Execution Accelerator** | **Qualcomm Hexagon v81 NPU** | **Qualcomm Hexagon v81 NPU** |
| **Runtime Provider** | `QNNExecutionProvider` (`QnnHtp.dll`) | `QNNExecutionProvider` (`QnnHtp.dll`) |
| **CPU Fallback** | **NO** (Pure NPU Execution) | **NO** (Pure NPU Execution) |
| **Input Shape** | Static `1x3x640x640` (float32) | Static `1x3x640x640` (float32) |
| **Average Latency** | **3.23 ms** | **5.55 ms** |
| **Median Latency** | **3.13 ms** | **5.45 ms** |
| **Minimum Latency** | **3.01 ms** | **4.82 ms** |
| **Maximum Latency** | **6.54 ms** | **32.63 ms** |
| **Processing Throughput** | **309.23 FPS** | **146.84 FPS** |
| **Tracking Overhead** | — | **0.00 ms** (Sub-millisecond) |
| **Threat Assessment Overhead** | — | **0.01 ms** (Sub-millisecond) |
| **Protection Manager Overhead** | — | **0.01 ms** (Sub-millisecond) |
| **CPU Utilization** | < 4% | **10.8%** |
| **Process Memory (RSS)** | 215.0 MB | **481.2 MB** |
| **Zero Network Calls** | **0 Observed** | **0 Observed** |

> [!NOTE]
> **Processing Throughput vs. Physical Camera Capture FPS:**
> The **146.84 FPS** and **309.23 FPS** figures represent benchmark **processing throughput** measured during full pipeline evaluation. They demonstrate the compute capacity of the Hexagon NPU. Physical webcam hardware captures video at its standard configured rate (~30 FPS).

---

## Real Threat Scenario Validation on NPU

All 7 security scenarios were evaluated on the real Snapdragon X2 Elite Hexagon NPU:

| Scenario | Conditions | Threat Score | Threat Level | Privacy Shield | Protection Action | Recovery Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **A. Primary User Only** | Primary user alone in workspace | **0.0** | `SAFE` | False | `NONE` | Baseline safe state |
| **B. Secondary Enters** | Secondary person enters room background | **25.0** | `SAFE` | False | `NONE` | Elevated monitoring |
| **C. Secondary Approaches** | Secondary walks towards desk ($>3\text{s}$) | **74.6** | `WARNING` | **True** | `WARNING_OVERLAY` | Amber warning shield activates |
| **D. Privacy Zone Breach** | Secondary enters workstation privacy zone | **75.0** | `WARNING` | **True** | `WARNING_OVERLAY` | Zone breach alert generated |
| **E. Screen-Capture Risk**| Phone detected near active screen | **100.0** | `CRITICAL` | **True** | `STRONG_OVERLAY` | Blackout shield deployed |
| **F. Primary Absent + Secondary**| Primary leaves while secondary present | **100.0** | `CRITICAL` | **True** | `STRONG_OVERLAY` | Immediate lockdown trigger |
| **G. Secondary Leaves** | Secondary leaves field of view | **0.0** | `SAFE` | False | `NONE` | Smooth recovery after hysteresis |

> [!IMPORTANT]
> **Technical Threat Model Terminology:**
> - Edge Sentinel assesses **potential physical privacy risks**, **shoulder-surfing risks**, **proximity risks**, and **privacy-zone breaches**.
> - Detection of a smartphone indicates a **potential screen-capture risk**; the system does **not** claim or infer that the device is actively recording.
> - The application identifies **primary user** vs. **secondary persons** via spatial/temporal proximity heuristics without facial recognition or biometric enrollment.

---

## Privacy by Design & Security Architecture

Designed for **zero-cloud video processing**:

```text
Camera Frame
    ↓ (in-memory buffer only — zero disk write)
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

- **0 Network Calls**: 0 network calls observed during the real-hardware validation benchmark.
- **Zero Raw Frame Writes**: Camera frames are processed strictly in RAM and never written to disk or persistent storage.
- **Zero Facial Recognition**: No biometric embeddings, face templates, or identity profiles are generated or stored.
- **Strictly Local Telemetry**: Only bounding-box metadata, risk scores, and accelerator runtime metrics are logged.

---

## Repository Structure

```text
edge-sentinel/
├── .gitignore                          # Protected from committing credentials, keys, or .pem files
├── README.md                           # Master architectural & benchmark documentation
├── START_EDGE_SENTINEL.bat              # One-click Windows launcher with priority venv detection
├── requirements.txt                    # Project dependencies (Desktop CPU vs. ARM64 Snapdragon)
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
│   │   ├── yolo_engine.py              # Lightweight PyTorch CPU inference engine
│   │   ├── snapdragon_engine.py        # Qualcomm QNN Hexagon NPU engine (Pillow + pure NumPy NMS)
│   │   └── validator.py                # Model compatibility & schema validation utility
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
│   │   ├── schema.py                   # PresenceState & WorkspaceContext data schemas
│   │   └── primary_user.py             # PrimaryUserEstimator & presence state machine
│   │
│   ├── threat/                         # Threat Assessment Layer
│   │   ├── __init__.py
│   │   ├── schema.py                   # ThreatLevel, PrivacyZone, and ThreatAssessment schemas
│   │   ├── engine.py                   # ThreatAssessmentEngine with debouncing & explainable reasons
│   │   └── device.py                   # DeviceThreatDetector for screen-capture device risks
│   │
│   ├── protection/                     # Active Privacy Defense Layer
│   │   ├── __init__.py
│   │   ├── schema.py                   # ProtectionAction & ShieldState schemas
│   │   ├── overlay.py                  # Non-destructive Windows privacy shield overlay
│   │   ├── lock.py                     # Native Windows WorkstationLocker guardrail
│   │   └── manager.py                  # ProtectionManager coordinating hysteresis & recovery
│   │
│   └── ui/                             # Presentation Layer
│       ├── __init__.py
│       └── dashboard.py                # Modern CustomTkinter dark-mode desktop GUI
│
├── models/                             # Model Storage & Compilation Tooling
│   ├── yolov8n.pt                      # PyTorch YOLOv8n base weights (6.2 MB)
│   └── qualcomm/
│       ├── README.md                   # Qualcomm Snapdragon compilation & licensing guide
│       ├── export_snapdragon.py        # Static ONNX graph export (1x3x640x640, opset 17)
│       └── yolov8n.onnx                # Static ONNX model for QNN HTP runtime (~12.2 MB)
│
└── tests/                              # Comprehensive Test Suite (42 Tests)
    ├── test_tracking_and_context.py    # Centroid tracking & primary presence tests (7 tests)
    ├── test_threat_engine.py           # Threat scoring, weights & debounce tests (12 tests)
    ├── test_phase4_protection_and_device.py # Device threat & privacy shield tests (12 tests)
    └── test_phase5_snapdragon_and_fallback.py # Snapdragon QNN engine & fallback tests (11 tests)
```

---

## Test Suite & Regression Validation

Edge Sentinel includes a test suite covering tracking, presence state transitions, threat weight scoring, device detection, privacy shield hysteresis, and Snapdragon QNN fallback mechanics:

```powershell
python -m pytest tests/ -q
..........................................                               [100%]
42 passed in 12.63s
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

#### Step C: Export Static ONNX Graph for QNN
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

```yaml
inference:
  backend: "auto"              # 'auto', 'snapdragon', or 'cpu'
  device: "cpu"                # Device for CPU backend ('cpu')
  confidence_threshold: 0.5    # Minimum detection confidence
  input_size: 640              # Model input dimension (640x640)

camera:
  device_index: 0              # Physical webcam index
  width: 640                   # Capture width
  height: 480                  # Capture height
  fps: 30                      # Camera target capture framerate

threat:
  weights:
    secondary_person: 25.0     # Baseline secondary person weight
    proximity: 20.0            # Proximity weight
    persistence: 15.0          # Persistence over time weight
    privacy_zone: 15.0         # Workstation privacy zone breach weight
    approach: 15.0             # Approach trajectory vector weight
    primary_absent: 25.0       # Primary user absence multiplier weight
    device_threat: 20.0        # Cell phone / capture risk weight
  thresholds:
    safe_max: 39.0             # SAFE threshold (0 - 39)
    warning_max: 79.0          # WARNING threshold (40 - 79)
    critical_min: 80.0         # CRITICAL threshold (80 - 100)
  hysteresis:
    smoothing_window: 5        # Number of frames for score debouncing
    deactivation_hold_s: 2.0   # Shield lock cooldown time in seconds

protection:
  allow_workstation_lock: false # Safety guardrail: native lock disabled by default
  lock_cooldown_s: 30.0        # Cooldown guardrail between lock events
```

---

## License & Third-Party Attributions

- **Edge Sentinel**: Distributed under the [MIT License](LICENSE).
- **YOLOv8**: Developed by [Ultralytics](https://github.com/ultralytics/ultralytics) under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html). Commercial deployments requiring proprietary terms must secure a commercial license directly from Ultralytics.
- **Qualcomm Snapdragon & Hexagon**: Registered trademarks of Qualcomm Incorporated. QNN SDK and Hexagon libraries are proprietary to Qualcomm Technologies, Inc.
