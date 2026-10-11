"""
evaluate.py - Evaluation routines and clinical metrics calculation
for Anemia (Regression) and Conjunctivitis (Classification).
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes clinical regression metrics for laboratory Hb prediction:
      - MAE (g/dL)
      - RMSE (g/dL)
      - R² score
      - Pearson correlation r
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    # Pearson correlation r
    if np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = 0.0
        
    return {
        'mae_g_dl': round(mae, 4),
        'rmse_g_dl': round(rmse, 4),
        'r2': round(r2, 4),
        'pearson_r': round(pearson_r, 4),
        'sample_count': len(y_true)
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Computes diagnostic classification metrics:
      - Accuracy, Precision, Recall/Sensitivity, Specificity, F1, ROC-AUC
      - Confusion Matrix
    """
    y_true = np.asarray(y_true, dtype=np.int64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.int64).flatten()
    y_prob = np.asarray(y_prob, dtype=np.float64).flatten()
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.0
        
    cm = confusion_matrix(y_true, y_pred).tolist()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(cm) == 2 else (0, 0, 0, 0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall_sensitivity': round(rec, 4),
        'specificity': round(spec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(auc, 4),
        'confusion_matrix': cm,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'sample_count': len(y_true)
    }
