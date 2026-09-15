"""Export YOLOv8n model for Qualcomm Snapdragon QNN compilation.

Prepares a static-graph ONNX model formatted specifically for Qualcomm QNN / AI Hub
compiler consumption, maintaining person (0) and cell phone (67) detection capabilities.
"""

import os
import argparse

def export_model(weights_path: str = "models/yolov8n.pt", output_dir: str = "models/qualcomm"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "yolov8n.onnx")

    print(f"[QualcommExport] Exporting '{weights_path}' for Qualcomm QNN compiler...")
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)

        # Export with static batch=1, imgsz=640, opset=17 for maximum QNN HTP compatibility
        exported_file = model.export(
            format="onnx",
            imgsz=640,
            batch=1,
            opset=17,
            simplify=True,
            dynamic=False
        )

        # Move/copy to qualcomm directory if needed
        if exported_file and os.path.exists(exported_file) and exported_file != out_path:
            import shutil
            shutil.copyfile(exported_file, out_path)

        print(f"[QualcommExport] Export successful: {out_path}")
        print("[QualcommExport] Ready for QNN conversion: qnn-onnx-converter or Qualcomm AI Hub CLI.")
        return out_path
    except Exception as e:
        print(f"[QualcommExport] Error exporting model: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8n for Qualcomm Snapdragon compilation")
    parser.add_argument("--weights", default="models/yolov8n.pt", help="Path to input PyTorch weights")
    parser.add_argument("--output_dir", default="models/qualcomm", help="Target output directory")
    args = parser.parse_args()
    export_model(args.weights, args.output_dir)
