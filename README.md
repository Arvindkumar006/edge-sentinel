# Edge Sentinel — On-Device AI Privacy Agent (Phase 5: Qualcomm Snapdragon AI Acceleration)

Edge Sentinel is a privacy-preserving AI security desktop application designed to monitor workstations for secondary persons, shoulder-surfing risks, and physical privacy risks using lightweight computer vision models running 100% locally.

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

### 2. Verified Real-Hardware Validation State

Edge Sentinel has been **fully validated on physical Qualcomm Snapdragon ARM64 hardware**:

- **Device**: Qualcomm Compute Reference Design SC8480XP / MTP
- **Processor**: Snapdragon X2 Elite X2E88100 (18 Oryon CPU cores, Hexagon v81 NPU)
- **Architecture**: Native ARM64 (aarch64), Windows 11 Enterprise
- **Runtime**: Python 3.12.9 ARM64, `onnxruntime-qnn==1.24.4`
- **Execution Provider**: `QNNExecutionProvider` with `QnnHtp.dll` backend
- **CPU Fallback**: **NO** (Active Hexagon NPU graph execution confirmed via VTCM allocation and DDR telemetry)
- **Test Suite**: **42 / 42 automated tests passed**

---

## Performance Benchmark Results

### Standalone Model vs. Complete Security Pipeline

| Metric | Standalone YOLOv8n NPU Inference | Complete Edge Sentinel Pipeline |
| :--- | :--- | :--- |
| **Hardware Platform** | Snapdragon X2 Elite X2E88100 | Snapdragon X2 Elite X2E88100 |
| **Execution Accelerator** | Qualcomm Hexagon v81 NPU | Qualcomm Hexagon v81 NPU |
| **Runtime Engine** | ONNX Runtime QNN (`QnnHtp.dll`) | ONNX Runtime QNN (`QnnHtp.dll`) |
| **CPU Fallback** | **No** | **No** |
| **Input Shape** | Static `1x3x640x640` (float32) | Static `1x3x640x640` (float32) |
| **Latency (Average)** | **3.23 ms** | **5.55 ms** |
| **Latency (Median)** | **3.13 ms** | **5.45 ms** |
| **Latency (Min / Max)** | **3.01 ms / 6.54 ms** | **4.82 ms / 32.63 ms** |
| **Processing Throughput** | **309.23 FPS** | **146.84 FPS** |
| **CPU Utilization** | < 4% | **10.8%** |
| **Process Memory (RSS)** | 215 MB | **481.2 MB** |
| **Regression Validation** | Verified | **42 / 42 tests passed** |

> [!NOTE]
> **Processing Throughput vs. Camera Capture FPS:**
> The **146.84 FPS** and **309.23 FPS** metrics represent maximum pipeline **processing throughput** measured during the benchmark evaluation loop. They do not represent the physical webcam hardware capture rate, which operates at its configured capture target (~30 FPS).

---

## Real Threat Scenario Validation on NPU

All 7 security scenarios were evaluated on the real Snapdragon X2 Elite Hexagon NPU:

| Scenario | Conditions | Threat Score | Threat Level | Privacy Shield | Protection Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Primary User Only** | Primary user alone in workspace | **0.0** | SAFE | False | None |
| **B. Secondary Enters** | Secondary person enters room background | **25.0** | SAFE | False | Monitoring elevated |
| **C. Secondary Approaches** | Secondary person walks towards desk | **74.6** | WARNING | True | WARNING_OVERLAY |
| **D. Privacy Zone Breach** | Secondary enters workstation privacy zone | **75.0** | WARNING | True | WARNING_OVERLAY |
| **E. Potential Screen-Capture Risk**| Phone detected near active screen | **100.0** | CRITICAL | True | STRONG_OVERLAY |
| **F. Primary Absent + Secondary** | Primary leaves workstation while secondary present | **100.0** | CRITICAL | True | STRONG_OVERLAY |
| **G. Secondary Leaves** | Secondary leaves field of view | **0.0** | SAFE | False | Recovery after hysteresis |

> [!IMPORTANT]
> **Technical Threat Model Terminology:**
> - Edge Sentinel assesses **potential physical privacy risks**, **shoulder-surfing risks**, **proximity risks**, and **privacy-zone breaches**.
> - Detection of a smartphone indicates a **potential screen-capture risk**; the system does **not** claim or infer that the device is actively recording.
> - The application identifies **primary user** vs. **secondary persons** via spatial/temporal proximity heuristics without facial recognition or biometric enrollment.

---

## Privacy Architecture & Guarantees

Designed for **zero-cloud video processing**:

```text
Camera Frame
    ↓ (in-memory buffer only)
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

## Reproducibility & Deployment Guide

### Target Hardware: Qualcomm Snapdragon ARM64 Windows 11 Copilot+ PC

1. **Verify Environment**:
   ```powershell
   python --version
   # Expected: Python 3.12.x (ARM64)

   python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
   # Expected: 1.24.4, ['QNNExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   pip install onnxruntime-qnn==1.24.4
   ```

3. **Export Static ONNX Graph for QNN**:
   ```powershell
   python models/qualcomm/export_snapdragon.py --weights models/yolov8n.pt --output_dir models/qualcomm
   ```

4. **Run Automated Test Suite (42 Tests)**:
   ```powershell
   pytest -q
   # Expected: 42 passed
   ```

5. **Run Hardware Benchmark**:
   ```powershell
   python main.py --benchmark --duration 30
   ```

6. **Launch Desktop Application**:
   ```powershell
   python main.py
   ```


