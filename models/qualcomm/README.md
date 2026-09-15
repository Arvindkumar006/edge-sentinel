# Qualcomm Snapdragon AI Acceleration Deployment Guide

This directory contains the model export tooling, runtime configuration, and deployment instructions for running Edge Sentinel's YOLOv8n detector on Qualcomm Snapdragon Copilot+ PCs (e.g. Snapdragon X2 Elite X2E88100 / Snapdragon X Elite / X Plus) powered by the Qualcomm Hexagon NPU.

---

## 1. Architectural Distinction & Execution Flow

```text
PyTorch YOLOv8n Model (.pt)
        ↓
Static ONNX Export (models/qualcomm/export_snapdragon.py: opset 17, static shape 1x3x640x640)
        ↓
QNN-compatible Snapdragon Runtime (models/qualcomm/yolov8n.onnx)
        ↓
ONNX Runtime QNNExecutionProvider (QnnHtp.dll backend)
        ↓
Qualcomm Hexagon HTP / NPU (Hardware Acceleration)
```

> [!CRITICAL]
> **Static ONNX Export vs. QNN Execution:**
> `models/qualcomm/export_snapdragon.py` exports a static-graph ONNX model with fixed dimensions (`1x3x640x640`, float32, opset 17) required by Qualcomm QNN. It does **not** perform standalone offline QNN binary compilation. Instead, the runtime compilation and NPU graph execution are performed on-device by ONNX Runtime's **`QNNExecutionProvider`** using Qualcomm's `QnnHtp.dll` backend.
>
> Edge Sentinel enforces strict truthfulness: it only reports `Accelerator: NPU` and `Status: ACTIVE` when `QNNExecutionProvider` is actively present in the ONNX Runtime session on physical hardware. Otherwise, it gracefully falls back to CPU (`Status: FALLBACK`).

---

## 2. Verified Hardware Environment

The deployment and NPU execution have been verified on:
- **Device**: Qualcomm Compute Reference Design SC8480XP / MTP
- **Processor**: Snapdragon X2 Elite X2E88100 (18 Oryon CPU cores, Hexagon v81 NPU)
- **OS**: Windows 11 Enterprise (ARM64)
- **Python**: Python 3.12.9 ARM64
- **ONNX Runtime**: `onnxruntime-qnn==1.24.4` (native `cp312-win_arm64`)
- **Backend Provider**: `QNNExecutionProvider` with `QnnHtp.dll`
- **CPU Fallback**: No (pure NPU execution confirmed via graph optimization and VTCM allocation telemetry)

---

## 3. Prerequisites for Snapdragon ARM64 Windows

On the target Snapdragon Copilot+ PC (ARM64 Windows 11):
1. **Qualcomm QNN / Hexagon Drivers**:
   - Ensure Qualcomm AI Engine Direct drivers (`QnnHtp.dll`, `QnnHtpPrepare.dll`, `QnnHtpV*.dll`) are installed with the system BSP or located in system PATH.
2. **ONNX Runtime with QNN Support**:
   ```bash
   pip install onnxruntime-qnn==1.24.4
   ```
   *(Note: `onnxruntime-qnn==1.24.4` includes the native Windows ARM64 provider for Hexagon NPU).*

---

## 4. Model Preparation Workflow

### Step 1: Export Static ONNX Graph
Run the export script to generate the static-shape ONNX model:
```bash
python models/qualcomm/export_snapdragon.py --weights models/yolov8n.pt --output_dir models/qualcomm
```
This generates `models/qualcomm/yolov8n.onnx` with:
- Static input shape: `1x3x640x640`
- Precision: `float32`
- Opset: `17`
- Size: ~12.2 MB

### Step 2: Runtime Execution via QNN Execution Provider
Edge Sentinel automatically initializes `QNNExecutionProvider` with HTP options:
```python
qnn_options = {
    "backend_path": "QnnHtp.dll",
    "htp_performance_mode": "burst",
    "enable_htp_fp16_precision": "1"
}
session = ort.InferenceSession("models/qualcomm/yolov8n.onnx", providers=[("QNNExecutionProvider", qnn_options), "CPUExecutionProvider"])
```

---

## 5. Model Licensing & Considerations

- **YOLOv8 Architecture**: YOLOv8 is developed by Ultralytics and distributed under the GNU Affero General Public License v3.0 (AGPL-3.0).
- **Commercial Licensing**: Any commercial deployment incorporating YOLOv8 weights or code must adhere to AGPL-3.0 copyleft terms or secure an enterprise commercial license from Ultralytics.
- **Model Scope**: Edge Sentinel filters YOLOv8 detections strictly for person (COCO class 0) and cell phone (COCO class 67). No biometric identification or facial recognition is performed.

