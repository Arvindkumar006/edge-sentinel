# Qualcomm Snapdragon AI Acceleration Deployment Guide

This directory contains the deployment workflow and compilation instructions for running Edge Sentinel's YOLOv8n model on Qualcomm Snapdragon Copilot+ PCs (Snapdragon X Elite / X Plus / 8cx Gen 3) powered by Qualcomm Hexagon NPU.

---

## 1. Architectural Distinction & Truthful Acceleration

```text
YOLOv8n PyTorch (.pt)
        ↓
ONNX Export (opset 17, static shape 1x3x640x640)
        ↓
QNN-compatible Conversion / Compilation (qnn-onnx-converter / Qualcomm AI Hub)
        ↓
Qualcomm QNN Context Binary (.bin / .dlc / QNN EP ONNX)
        ↓
ONNX Runtime QNNExecutionProvider (QnnHtp.dll)
        ↓
Verified Snapdragon Hexagon NPU
```

> [!CRITICAL]
> **Exporting to standard ONNX is NOT NPU execution.**
> Standard ONNX runs on CPU unless a specialized Execution Provider compiles the graph to NPU machine code. Edge Sentinel will only report `Accelerator: NPU` when `QNNExecutionProvider` with the Hexagon Tensor Processor (`QnnHtp.dll`) is verified at runtime. Otherwise, it gracefully falls back to CPU.

---

## 2. Prerequisites for Snapdragon Windows PC

On the target Snapdragon Copilot+ PC (ARM64 Windows 11):
1. **Qualcomm Neural Processing SDK / QNN SDK** (v2.20 or newer):
   - Install Qualcomm AI Engine Direct SDK (QNN SDK).
   - Ensure `QnnHtp.dll`, `QnnHtpPrepare.dll`, `QnnHtpV73Stub.dll` (or matching HTP architecture) are in the system `PATH`.
2. **ONNX Runtime QNN Package**:
   ```bash
   pip install onnxruntime-qnn
   ```
3. **Qualcomm AI Hub Client** (Alternative direct cloud compilation):
   ```bash
   pip install qai_hub
   qai-hub configure --api_token <YOUR_QUALCOMM_AI_HUB_TOKEN>
   ```

---

## 3. Model Compilation Workflow

### Option A: Official Qualcomm AI Hub Path (Recommended)
Compile directly for Snapdragon X Elite Hexagon NPU:
```bash
# 1. Export YOLOv8 to torchscript/onnx
python export_snapdragon.py --format onnx

# 2. Compile via Qualcomm AI Hub CLI for Snapdragon X Elite
qai-hub compile \
    --device "Snapdragon X Elite CRD" \
    --model "models/qualcomm/yolov8n.onnx" \
    --output-path "models/qualcomm/yolov8n_qnn.bin" \
    --options "--target_arch hexagon_v73"
```

### Option B: Local Qualcomm QNN Converter
Using the Qualcomm QNN SDK tools:
```bash
# Convert ONNX to QNN model
qnn-onnx-converter \
    --input_network models/qualcomm/yolov8n.onnx \
    --output_path models/qualcomm/yolov8n_qnn.cpp

# Compile into HTP context binary
qnn-model-lib-generator \
    -c models/qualcomm/yolov8n_qnn.cpp \
    -b models/qualcomm/yolov8n_qnn.bin \
    -t aarch64-windows-msvc
```

---

## 4. Edge Sentinel Configuration

In `config/config.yaml`:
```yaml
inference:
  backend: auto   # 'auto', 'snapdragon', or 'cpu'

model:
  qnn_model_path: "models/qualcomm/yolov8n.onnx"
  engine: "yolov8"
  input_size: 640
  target_classes: [0, 67]  # 0: person, 67: cell phone
```

- When set to `auto`: Edge Sentinel detects if Qualcomm NPU hardware and runtime are present. If verified, it activates `SNAPDRAGON (NPU)`. If not present (e.g. running on AMD64/x86 dev machine), it safely activates CPU fallback without interrupting the security pipeline.
