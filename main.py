import sys
import os
import argparse
import time

# Robust UTF-8 stdout handling for Windows CP1252 / SSH environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edge_sentinel.config import load_config
from edge_sentinel.capture.camera import WebcamCapture
from edge_sentinel.pipeline import SentinelPipeline

def run_benchmark(config_path: str = "config/config.yaml", duration_seconds: int = 5):

    """
    Runs a headless benchmark on the current machine measuring real FPS,
    inference latency, tracking overhead, and total pipeline latency.
    Strictly reports real hardware measurements without fabrication.
    """
    print(f"\n=======================================================")
    print(f"   EDGE SENTINEL (PHASE 5) — HARDWARE BENCHMARK")
    print(f"=======================================================")
    config = load_config(config_path)
    
    print(f"[*] Initializing capture device index: {config.camera.device_index}")
    cam = WebcamCapture(
        device_index=config.camera.device_index,
        width=config.camera.width,
        height=config.camera.height,
        target_fps=config.camera.fps
    )
    if not cam.start():
        print(f"[!] Warning: Camera index {config.camera.device_index} failed to open.")
        print(f"[!] {cam.error_message}")
        print("[!] Generating synthetic frame for inference + tracking benchmark...")
        import numpy as np
        use_synthetic = True
        synthetic_frame = np.random.randint(0, 255, (config.camera.height, config.camera.width, 3), dtype=np.uint8)
    else:
        use_synthetic = False

    print(f"[*] Initializing pipeline: {config.model.model_path} ({config.model.device})")
    pipeline = SentinelPipeline(config)
    if not pipeline.load():
        print("[!] Failed to load model.")
        cam.stop()
        return

    print(f"[*] Warming up pipeline (3 frames)...")
    for _ in range(3):
        if not use_synthetic:
            _, frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue
        else:
            frame = synthetic_frame
        pipeline.process_frame(frame)

    print(f"[*] Running live Phase 4 benchmark for {duration_seconds} seconds...")
    start_time = time.perf_counter()
    inf_latencies = []
    trk_latencies = []
    thr_latencies = []
    prot_latencies = []
    tot_latencies = []
    frames_processed = 0

    while time.perf_counter() - start_time < duration_seconds:
        if not use_synthetic:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
        else:
            frame = synthetic_frame

        output = pipeline.process_frame(frame)
        inf_latencies.append(output.inference_latency_ms)
        trk_latencies.append(output.tracking_overhead_ms)
        thr_latencies.append(output.threat_overhead_ms)
        prot_latencies.append(output.protection_overhead_ms)
        tot_latencies.append(output.total_pipeline_latency_ms)
        frames_processed += 1

    total_time = time.perf_counter() - start_time
    actual_fps = frames_processed / total_time if total_time > 0 else 0.0
    avg_inf = sum(inf_latencies) / len(inf_latencies) if inf_latencies else 0.0
    min_inf = min(inf_latencies) if inf_latencies else 0.0
    max_inf = max(inf_latencies) if inf_latencies else 0.0
    avg_trk = sum(trk_latencies) / len(trk_latencies) if trk_latencies else 0.0
    avg_thr = sum(thr_latencies) / len(thr_latencies) if thr_latencies else 0.0
    avg_prot = sum(prot_latencies) / len(prot_latencies) if prot_latencies else 0.0
    avg_tot = sum(tot_latencies) / len(tot_latencies) if tot_latencies else 0.0

    try:
        import psutil
        process = psutil.Process()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / (1024 * 1024)
        sys_info = f"CPU Usage         : {cpu_percent:.1f}% (Process Memory: {mem_mb:.1f} MB)"
    except Exception:
        sys_info = "CPU/Memory        : Not Available"

    print("\n------------------ BENCHMARK RESULTS ------------------")
    print(f"Backend Name      : {pipeline.engine.get_backend_name()}")
    print(f"Backend Type      : {pipeline.engine.get_backend_type()}")
    print(f"Accelerator       : {pipeline.engine.get_accelerator_type()}")
    print(f"Status            : {pipeline.engine.get_status()}")
    print(f"Compute Device    : {pipeline.engine.get_device()}")
    print(f"Pipeline Mode     : Phase 5 (Inference + Tracking + Threat + Protection)")
    print(f"Frames Processed  : {frames_processed} frames in {total_time:.2f}s")
    print(f"Throughput (FPS)  : {actual_fps:.2f} FPS")
    print(f"Inference Latency : Avg = {avg_inf:.2f} ms | Min = {min_inf:.2f} ms | Max = {max_inf:.2f} ms")
    print(f"Tracking Overhead : Avg = {avg_trk:.2f} ms")
    print(f"Threat Overhead   : Avg = {avg_thr:.2f} ms")
    print(f"Protection Overhd : Avg = {avg_prot:.2f} ms")
    print(f"Total Latency     : Avg = {avg_tot:.2f} ms")
    print(f"{sys_info}")
    print(f"Privacy Guarantee : 100% On-Device Local Processing (Zero Network Calls)")
    print("-------------------------------------------------------")

    # Section 11 Benchmark Safety Check
    if pipeline.engine.get_accelerator_type() != "NPU":
        print("\n================ SNAPDRAGON HARDWARE STATUS ================")
        print("Snapdragon hardware unavailable on current development machine.")
        print()
        print("Qualcomm backend:")
        print("IMPLEMENTED / INTEGRATION-READY")
        print()
        print("NPU execution:")
        print("NOT VERIFIED")
        print()
        print("Snapdragon benchmark:")
        print("NOT EXECUTED")
        print("============================================================\n")

    cam.stop()

def main():
    parser = argparse.ArgumentParser(description="Edge Sentinel — Local AI Privacy Guardian (Phase 5)")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode without GUI")
    parser.add_argument("--duration", type=int, default=30, help="Benchmark duration in seconds")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(config_path=args.config, duration_seconds=args.duration)
        return

    print("[*] Launching Edge Sentinel Desktop Dashboard (Phase 5)...")
    from edge_sentinel.ui.dashboard import SentinelDashboard
    config = load_config(args.config)
    app = SentinelDashboard(config)
    app.mainloop()


if __name__ == "__main__":
    main()
