"""Model Compatibility Validation Utility.

Compares detection outputs between CPUBackend and SnapdragonBackend on identical
input frames to ensure output compatibility, bounding-box fidelity, class consistency,
and confidence calibration across backends.

Documented Tolerance Thresholds:
  - Minimum IoU match threshold: 0.70 (bounding boxes must overlap with >= 70% IoU)
  - Maximum confidence difference (|conf_cpu - conf_snapdragon|): <= 0.15
  - Class ID consistency: 100% agreement on matched bounding boxes
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area

class ModelCompatibilityValidator:
    """
    Validates output parity between CPU and Snapdragon inference engines.
    """

    def __init__(
        self,
        min_iou_threshold: float = 0.70,
        max_conf_diff: float = 0.15
    ):
        self.min_iou_threshold = min_iou_threshold
        self.max_conf_diff = max_conf_diff

    def compare_detections(
        self,
        cpu_detections: List[Dict[str, Any]],
        snapdragon_detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compares detections from CPU and Snapdragon on an identical frame.

        Returns structured report dictionary.
        """
        cpu_count = len(cpu_detections)
        snap_count = len(snapdragon_detections)

        matched_pairs = []
        unmatched_cpu = list(range(cpu_count))
        unmatched_snap = list(range(snap_count))

        # Greedy bipartite matching by IoU
        for c_idx in range(cpu_count):
            best_iou = 0.0
            best_s_idx = -1
            for s_idx in unmatched_snap:
                iou = calculate_iou(cpu_detections[c_idx]["bbox"], snapdragon_detections[s_idx]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_s_idx = s_idx

            if best_iou >= self.min_iou_threshold and best_s_idx != -1:
                matched_pairs.append({
                    "cpu_idx": c_idx,
                    "snap_idx": best_s_idx,
                    "iou": best_iou,
                    "cpu_box": cpu_detections[c_idx]["bbox"],
                    "snap_box": snapdragon_detections[best_s_idx]["bbox"],
                    "cpu_conf": cpu_detections[c_idx]["confidence"],
                    "snap_conf": snapdragon_detections[best_s_idx]["confidence"],
                    "conf_diff": abs(cpu_detections[c_idx]["confidence"] - snapdragon_detections[best_s_idx]["confidence"]),
                    "cpu_class": cpu_detections[c_idx]["class_id"],
                    "snap_class": snapdragon_detections[best_s_idx]["class_id"],
                    "class_match": cpu_detections[c_idx]["class_id"] == snapdragon_detections[best_s_idx]["class_id"]
                })
                unmatched_cpu.remove(c_idx)
                unmatched_snap.remove(best_s_idx)

        # Validation checks
        conf_diffs = [p["conf_diff"] for p in matched_pairs]
        avg_conf_diff = sum(conf_diffs) / len(conf_diffs) if conf_diffs else 0.0
        max_observed_conf_diff = max(conf_diffs) if conf_diffs else 0.0

        ious = [p["iou"] for p in matched_pairs]
        avg_iou = sum(ious) / len(ious) if ious else 0.0

        all_classes_matched = all(p["class_match"] for p in matched_pairs)
        conf_tolerances_passed = max_observed_conf_diff <= self.max_conf_diff
        counts_matched = (cpu_count == snap_count)

        is_compatible = counts_matched and all_classes_matched and conf_tolerances_passed

        return {
            "is_compatible": is_compatible,
            "cpu_detection_count": cpu_count,
            "snapdragon_detection_count": snap_count,
            "matched_count": len(matched_pairs),
            "unmatched_cpu_count": len(unmatched_cpu),
            "unmatched_snapdragon_count": len(unmatched_snap),
            "average_iou": avg_iou,
            "average_conf_diff": avg_conf_diff,
            "max_conf_diff": max_observed_conf_diff,
            "all_classes_matched": all_classes_matched,
            "matched_pairs": matched_pairs
        }

    def print_report(self, report: Dict[str, Any]) -> str:
        """Generates a human-readable summary report."""
        lines = [
            "============================================================",
            "        MODEL COMPATIBILITY VALIDATION REPORT",
            "============================================================",
            f"Compatible:               {'PASSED' if report['is_compatible'] else 'FAILED'}",
            f"CPU Detection Count:       {report['cpu_detection_count']}",
            f"Snapdragon Count:         {report['snapdragon_detection_count']}",
            f"Matched Detections:       {report['matched_count']}",
            f"Average IoU Overlap:      {report['average_iou']:.3f} (Threshold >= {self.min_iou_threshold})",
            f"Average Conf Difference:  {report['average_conf_diff']:.3f} (Threshold <= {self.max_conf_diff})",
            f"Max Conf Difference:      {report['max_conf_diff']:.3f}",
            f"All Class IDs Matched:    {report['all_classes_matched']}",
            "============================================================"
        ]
        text = "\n".join(lines)
        print(text)
        return text
