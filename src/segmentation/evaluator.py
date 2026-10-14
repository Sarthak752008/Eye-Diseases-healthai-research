"""
src/segmentation/evaluator.py
------------------------------
Evaluates the ocular segmenter on the Eye Conjunctiva Segmentation Dataset.
Computes Dice coefficient, Jaccard Index (IoU), precision, recall,
and inter-annotator baseline comparison across dual ground-truth masks.
"""

import os
import glob
import time
from typing import Dict, Any, List
import numpy as np
import cv2
from PIL import Image

from src.segmentation.clinical_segmenter import ClinicalOcularSegmenter


def evaluate_segmentation_benchmark(
    dataset_root: str,
    max_samples: int = 50,
    stride: int = 10
) -> Dict[str, Any]:
    """
    Runs segmentation evaluation across annotated images.
    Compares segmenter predictions against Annotator 1 and Annotator 2 ground truths.
    """
    img_dir = os.path.join(dataset_root, "Images")
    m1_dir = os.path.join(dataset_root, "Masks Annotator 1")
    m2_dir = os.path.join(dataset_root, "Masks Annotator 2")

    if not os.path.exists(img_dir) or not os.path.exists(m1_dir):
        return {
            "status": "dataset_missing",
            "message": f"Dataset directory not found at: {dataset_root}",
            "sample_count": 0
        }

    all_images = sorted(glob.glob(os.path.join(img_dir, "*.png")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]) if os.path.splitext(os.path.basename(p))[0].isdigit() else 0)
    selected_images = all_images[::stride][:max_samples]

    segmenter = ClinicalOcularSegmenter()
    results = []

    for img_path in selected_images:
        fname = os.path.basename(img_path)
        m1_path = os.path.join(m1_dir, fname)
        m2_path = os.path.join(m2_dir, fname)

        if not os.path.exists(m1_path):
            continue

        gt1 = cv2.imread(m1_path, cv2.IMREAD_GRAYSCALE)
        if gt1 is None:
            continue
        gt1_bool = (gt1 > 0)

        gt2_bool = None
        if os.path.exists(m2_path):
            gt2 = cv2.imread(m2_path, cv2.IMREAD_GRAYSCALE)
            if gt2 is not None:
                gt2_bool = (gt2 > 0)

        pil_img = Image.open(img_path)
        seg_res = segmenter.segment(pil_img)
        pred_mask = seg_res.conjunctiva_mask

        # Metrics vs Annotator 1
        inter1 = np.logical_and(pred_mask, gt1_bool).sum()
        union1 = np.logical_or(pred_mask, gt1_bool).sum()
        dice1 = (2.0 * inter1) / (pred_mask.sum() + gt1_bool.sum() + 1e-6)
        iou1 = inter1 / (union1 + 1e-6)

        # Precision & Recall
        prec1 = inter1 / (pred_mask.sum() + 1e-6)
        rec1 = inter1 / (gt1_bool.sum() + 1e-6)

        entry = {
            "file": fname,
            "dice_annotator1": round(float(dice1), 4),
            "iou_annotator1": round(float(iou1), 4),
            "precision": round(float(prec1), 4),
            "recall": round(float(rec1), 4),
            "coverage_pct": seg_res.coverage_pct["conjunctiva"]
        }

        # Inter-annotator baseline if Annotator 2 present
        if gt2_bool is not None:
            inter12 = np.logical_and(gt1_bool, gt2_bool).sum()
            union12 = np.logical_or(gt1_bool, gt2_bool).sum()
            dice12 = (2.0 * inter12) / (gt1_bool.sum() + gt2_bool.sum() + 1e-6)
            iou12 = inter12 / (union12 + 1e-6)
            entry["annotator1_vs_2_dice"] = round(float(dice12), 4)
            entry["annotator1_vs_2_iou"] = round(float(iou12), 4)

        results.append(entry)

    if not results:
        return {"status": "no_evaluated_samples", "sample_count": 0}

    mean_dice = round(float(np.mean([r["dice_annotator1"] for r in results])), 4)
    mean_iou = round(float(np.mean([r["iou_annotator1"] for r in results])), 4)
    mean_prec = round(float(np.mean([r["precision"] for r in results])), 4)
    mean_rec = round(float(np.mean([r["recall"] for r in results])), 4)

    summary = {
        "status": "success",
        "sample_count": len(results),
        "mean_dice_score": mean_dice,
        "mean_iou_score": mean_iou,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "model_info": segmenter.get_model_info(),
        "individual_results": results[:10]  # First 10 samples
    }

    if "annotator1_vs_2_dice" in results[0]:
        summary["inter_annotator_mean_dice"] = round(float(np.mean([r["annotator1_vs_2_dice"] for r in results])), 4)
        summary["inter_annotator_mean_iou"] = round(float(np.mean([r["annotator1_vs_2_iou"] for r in results])), 4)

    return summary


if __name__ == "__main__":
    import json
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Eye Conjunctiva Segmentation Dataset"))
    print("Running Ocular Segmentation Benchmark...")
    res = evaluate_segmentation_benchmark(dataset_path, max_samples=25, stride=20)
    print(json.dumps(res, indent=2))
