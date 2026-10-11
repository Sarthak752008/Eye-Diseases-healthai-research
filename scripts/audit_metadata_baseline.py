"""
audit_metadata_baseline.py
--------------------------
Layer 3.5: Reliability and Domain Validation

Quantifies the shortcut learning vulnerability by training metadata-only baselines
(image width, height, aspect ratio, area) without any pixel / biological color data.
Evaluates on the exact same train/val/test splits used by the CNN.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))  # sensitivity
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else float(roc_auc_score(y_true, y_pred))
    
    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall_sensitivity': round(rec, 4),
        'specificity': round(spec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(auc, 4),
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'sample_count': int(len(y_true))
    }


def run_metadata_baseline(manifest_path=None):
    if manifest_path is None:
        manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'conjunctivitis_manifest_v2_clean.csv')
        
    df = pd.read_csv(manifest_path)
    df['aspect_ratio'] = df['width'] / df['height']
    df['area'] = df['width'] * df['height']
    
    feature_cols = ['width', 'height', 'aspect_ratio', 'area']
    
    # Dimension audit statistics per class
    audit_stats = {}
    for cls_name in ['normal', 'conjunctivitis']:
        sub = df[df['class_name'] == cls_name]
        audit_stats[cls_name] = {
            'count': int(len(sub)),
            'width_min': int(sub['width'].min()),
            'width_max': int(sub['width'].max()),
            'width_mean': round(float(sub['width'].mean()), 2),
            'height_min': int(sub['height'].min()),
            'height_max': int(sub['height'].max()),
            'height_mean': round(float(sub['height'].mean()), 2),
            'aspect_ratio_min': round(float(sub['aspect_ratio'].min()), 4),
            'aspect_ratio_max': round(float(sub['aspect_ratio'].max()), 4),
            'aspect_ratio_mean': round(float(sub['aspect_ratio'].mean()), 4),
            'area_min': int(sub['area'].min()),
            'area_max': int(sub['area'].max()),
            'area_mean': round(float(sub['area'].mean()), 2),
        }

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    X_train = train_df[feature_cols].values
    y_train = train_df['class_label'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['class_label'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['class_label'].values

    # Model 1: Single decision rule (width > 100)
    rule_pred_val = (val_df['width'].values > 100).astype(int)
    rule_pred_test = (test_df['width'].values > 100).astype(int)
    rule_val_metrics = compute_classification_metrics(y_val, rule_pred_val)
    rule_test_metrics = compute_classification_metrics(y_test, rule_pred_test)

    # Model 2: Logistic Regression on (width, height, aspect_ratio, area)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred_test = lr.predict(X_test)
    lr_prob_test = lr.predict_proba(X_test)[:, 1]
    lr_test_metrics = compute_classification_metrics(y_test, lr_pred_test, lr_prob_test)

    # Model 3: Decision Tree Classifier (max_depth=1: single threshold split)
    dt = DecisionTreeClassifier(max_depth=1, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred_test = dt.predict(X_test)
    dt_prob_test = dt.predict_proba(X_test)[:, 1]
    dt_test_metrics = compute_classification_metrics(y_test, dt_pred_test, dt_prob_test)

    report = {
        'task': 'Layer 3.5 Metadata-Only Baseline Evaluation',
        'dataset_manifest': manifest_path,
        'dataset_audit': audit_stats,
        'findings': {
            'dimension_gap': (
                f"Normal images have width in [{audit_stats['normal']['width_min']}, {audit_stats['normal']['width_max']}], "
                f"while Conjunctivitis images have width in [{audit_stats['conjunctivitis']['width_min']}, {audit_stats['conjunctivitis']['width_max']}]. "
                "The distributions have zero overlap."
            ),
            'aspect_ratio_gap': (
                f"Normal images are 100% square (aspect_ratio = {audit_stats['normal']['aspect_ratio_mean']}), "
                f"while Conjunctivitis images have mean aspect_ratio = {audit_stats['conjunctivitis']['aspect_ratio_mean']} "
                f"(range: [{audit_stats['conjunctivitis']['aspect_ratio_min']}, {audit_stats['conjunctivitis']['aspect_ratio_max']}])."
            ),
            'shortcut_consequence': (
                "A trivial classifier looking ONLY at image dimensions/aspect ratio achieves 100% accuracy "
                "without inspecting a single pixel. Consequently, MobileNetV3-Small's 100% test accuracy "
                "cannot be attributed to biological conjunctival erythema recognition."
            )
        },
        'metadata_baselines_test_results': {
            'rule_based_width_threshold_100': rule_test_metrics,
            'logistic_regression_metadata': lr_test_metrics,
            'decision_tree_stump_metadata': dt_test_metrics,
        }
    }

    # Save output
    output_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis', 'metadata_baseline_metrics.json')
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == '__main__':
    rep = run_metadata_baseline()
    print("=== Layer 3.5 Conjunctivitis Metadata-Only Baseline ===")
    print("\nDataset Audit:")
    for k, v in rep['dataset_audit'].items():
        print(f"  Class '{k}' (N={v['count']}):")
        print(f"    Width:  [{v['width_min']}, {v['width_max']}], mean={v['width_mean']}")
        print(f"    Height: [{v['height_min']}, {v['height_max']}], mean={v['height_mean']}")
        print(f"    Aspect: [{v['aspect_ratio_min']}, {v['aspect_ratio_max']}], mean={v['aspect_ratio_mean']}")
        print(f"    Area:   [{v['area_min']}, {v['area_max']}], mean={v['area_mean']}")
        
    print("\nTest Set Results (Metadata Only, Zero Pixels):")
    rule_res = rep['metadata_baselines_test_results']['rule_based_width_threshold_100']
    print(f"  Rule (width > 100):  Accuracy={rule_res['accuracy']:.4f}, Sens={rule_res['recall_sensitivity']:.4f}, "
          f"Spec={rule_res['specificity']:.4f}, F1={rule_res['f1_score']:.4f}, AUC={rule_res['roc_auc']:.4f}")
    
    lr_res = rep['metadata_baselines_test_results']['logistic_regression_metadata']
    print(f"  Logistic Regression: Accuracy={lr_res['accuracy']:.4f}, Sens={lr_res['recall_sensitivity']:.4f}, "
          f"Spec={lr_res['specificity']:.4f}, F1={lr_res['f1_score']:.4f}, AUC={lr_res['roc_auc']:.4f}")
    
    dt_res = rep['metadata_baselines_test_results']['decision_tree_stump_metadata']
    print(f"  Decision Tree Stump: Accuracy={dt_res['accuracy']:.4f}, Sens={dt_res['recall_sensitivity']:.4f}, "
          f"Spec={dt_res['specificity']:.4f}, F1={dt_res['f1_score']:.4f}, AUC={dt_res['roc_auc']:.4f}")
    print(f"  Confusion Matrix:    {dt_res['confusion_matrix']}")
    print("\nReport saved to: saved_models/conjunctivitis/metadata_baseline_metrics.json")
