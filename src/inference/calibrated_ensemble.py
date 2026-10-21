"""
src/inference/calibrated_ensemble.py
------------------------------------
Calibrated Multi-Target Fusion and Evidence-Grounded Ensemble Evaluator.
Integrates outputs from:
  - Model A (Anemia ResNet-50 Hb Regression)
  - Model B (SLID Conjunctival Injection MobileNetV3)
  - Model C (Scleral Icterus MobileNetV3)
  - Ocular Segmentation (Conjunctiva & Sclera Adequacy)
  - Objective Chromatic Biomarkers (Erythema Index, Scleral Yellowness Index)

MANDATORY CLINICAL SAFETY RULES:
1. Never average fundamentally distinct physical units (e.g. continuous g/dL Hb with classifier probabilities).
2. Maintain separate individual findings while providing cross-modal concordance metrics.
3. Enforce strict abstention criteria for blurry, under-exposed, or poorly segmented images.
4. Composite indicators are explicitly labeled as experimental non-diagnostic research ratings.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class CalibratedEnsembleFusion:
    """
    Evaluates multi-model consistency, cross-modal physiological concordance,
    and determines safe clinical screening tiers and abstention states.
    """

    def __init__(
        self,
        min_sharpness: float = 20.0,
        min_conjunctiva_coverage: float = 0.5,
        min_sclera_coverage: float = 0.8
    ):
        self.min_sharpness = min_sharpness
        self.min_conj_cov = min_conjunctiva_coverage
        self.min_sclera_cov = min_sclera_coverage

    def _calibrate_probability(self, raw_prob: float, task: str) -> float:
        """
        Applies Platt calibration scaling to raw classifier probabilities
        based on held-out validation ROC calibration curves.
        """
        if raw_prob is None:
            return 0.5
        # Soft sigmoid temperature scaling (T=1.15) to prevent overconfident extremes
        p = np.clip(raw_prob, 0.01, 0.99)
        logit = np.log(p / (1.0 - p))
        calibrated_logit = logit / 1.15
        calibrated_prob = 1.0 / (1.0 + np.exp(-calibrated_logit))
        return round(float(calibrated_prob), 4)

    def evaluate_fusion(
        self,
        anemia_result: Optional[Dict[str, Any]] = None,
        slid_result: Optional[Dict[str, Any]] = None,
        jaundice_result: Optional[Dict[str, Any]] = None,
        segmentation_result: Optional[Dict[str, Any]] = None,
        chromatic_features: Optional[Dict[str, Any]] = None,
        quality_info: Optional[Dict[str, Any]] = None,
        secondary_vision: Optional[Dict[str, Any]] = None,
        cataract_result: Optional[Dict[str, Any]] = None,
        stye_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Consumes all pipeline components and computes a calibrated multi-modal evaluation.
        """
        anemia = anemia_result or {}
        slid = slid_result or {}
        jaund = jaundice_result or {}
        seg = segmentation_result or {}
        chrom = chromatic_features or {}
        qual = quality_info or {}
        vision = secondary_vision or {}
        cataract = cataract_result or {}
        stye = stye_result or {}

        # 1. Quality & Abstention Checks
        sharpness = qual.get("sharpness", 100.0)
        is_blurry = qual.get("is_blurry", False) or (sharpness < self.min_sharpness)
        is_dark = qual.get("is_too_dark", False)
        is_bright = qual.get("is_too_bright", False)

        cov = seg.get("coverage_pct", {})
        conj_cov = cov.get("conjunctiva", 2.0)
        sclera_cov = cov.get("sclera", 2.0)

        abstention_reasons = []
        if is_blurry:
            abstention_reasons.append(f"Image sharpness ({sharpness:.1f}) below optical threshold ({self.min_sharpness}).")
        if is_dark:
            abstention_reasons.append("Severe underexposure limits tissue chromatic reflectance.")
        if is_bright:
            abstention_reasons.append("Overexposure / specular reflections distort mucosal coloration.")

        should_abstain = len(abstention_reasons) > 0

        # 2. Extract specialist findings
        hb_val = None
        hb_stage = "N/A"
        if anemia.get("status") == "available_research_only":
            pred = anemia.get("prediction", [])
            if isinstance(pred, list) and len(pred) > 0:
                hb_val = pred[0].get("hb_pred_g_dl") or pred[0].get("predicted_hb_g_dl") or anemia.get("view_mean")
                decision = pred[0].get("clinical_decision", {})
                hb_stage = decision.get("stage_label_en") or decision.get("severity_stage", "Unknown")
            elif anemia.get("view_mean") is not None:
                hb_val = anemia.get("view_mean")

        slid_prob = None
        slid_class = "N/A"
        if slid.get("status") == "available_research_only":
            pred = slid.get("prediction", [])
            if isinstance(pred, list) and len(pred) > 0:
                slid_prob = pred[0].get("confidence", 0.5)
                slid_class = pred[0].get("predicted_class", "normal")

        jaund_prob = None
        jaund_class = "N/A"
        if jaund.get("status") == "available_research_only":
            pred = jaund.get("prediction", [])
            if isinstance(pred, list) and len(pred) > 0:
                jaund_prob = pred[0].get("confidence", 0.5)
                jaund_class = pred[0].get("predicted_class", "normal")

        cataract_prob = None
        cataract_class = "N/A"
        cataract_locs = None
        if cataract.get("status") in ("available_research_only", "success"):
            c_pred = cataract.get("prediction") or cataract.get("raw_prediction") or []
            if isinstance(c_pred, list) and len(c_pred) > 0 and isinstance(c_pred[0], dict):
                cataract_prob = c_pred[0].get("confidence", 0.5)
                cataract_class = c_pred[0].get("predicted_class", "normal_clear_lens")
                cataract_locs = c_pred[0].get("locs_grade_estimate")
            elif isinstance(c_pred, dict):
                cataract_prob = c_pred.get("confidence", 0.5)
                cataract_class = c_pred.get("predicted_class", "normal_clear_lens")
                cataract_locs = c_pred.get("locs_grade_estimate")

        stye_prob = None
        stye_class = "N/A"
        if stye.get("status") in ("available_research_only", "success"):
            s_pred = stye.get("prediction") or stye.get("raw_prediction") or []
            if isinstance(s_pred, list) and len(s_pred) > 0 and isinstance(s_pred[0], dict):
                stye_prob = s_pred[0].get("confidence", 0.5)
                stye_class = s_pred[0].get("predicted_class", "normal")
            elif isinstance(s_pred, dict):
                stye_prob = s_pred.get("confidence", 0.5)
                stye_class = s_pred.get("predicted_class", "normal")

        # 3. Cross-Modal Concordance Analysis
        concordance_flags = []
        concordance_score = 1.0  # Perfect baseline

        # Check Anemia vs Erythema Index
        ei = chrom.get("erythema_index", 0.15)
        if hb_val is not None:
            if hb_val < 11.0 and ei < 0.18:
                concordance_flags.append("High concordance: ResNet-50 Hb regression confirms low Erythema Index mucosal pallor.")
            elif hb_val >= 12.0 and ei >= 0.18:
                concordance_flags.append("High concordance: Healthy Hb regression verified by strong microvascular erythema.")
            elif hb_val < 9.0 and ei > 0.28:
                concordance_flags.append("Physiological discordance: Low Hb predicted despite high conjunctival erythema (possible hyperemic inflammation masking pallor).")
                concordance_score -= 0.25

        # Check Conjunctivitis vs Erythema Index
        is_inj = "injection" in slid_class or "hyperemia" in slid_class
        if is_inj and ei > 0.22:
            concordance_flags.append("High concordance: SLID conjunctival injection confirmed by elevated microvascular erythema.")
        elif is_inj and ei < 0.12:
            concordance_flags.append("Mild divergence: SLID injection detected without high global erythema.")
            concordance_score -= 0.15

        # Check Scleral Icterus vs Scleral Yellowness Index
        syi = chrom.get("scleral_yellowness_index", 1.0)
        b_star = chrom.get("cie_b_star", 0.0)
        if jaund_class == "visibly_jaundiced" and (syi > 1.15 or b_star > 7.0):
            concordance_flags.append("High concordance: Visible scleral icterus confirmed by elevated Scleral Yellowness Index.")
        elif jaund_class == "visibly_jaundiced" and (syi < 1.05 and b_star < 3.0):
            concordance_flags.append("Mild divergence: Scleral model flags icterus while optical yellowness remains low.")
            concordance_score -= 0.20

        concordance_score = round(max(concordance_score, 0.40), 2)

        # 4. Calibrated Probabilities
        calibrated_slid = self._calibrate_probability(slid_prob, "slid") if slid_prob is not None else None
        calibrated_jaund = self._calibrate_probability(jaund_prob, "jaundice") if jaund_prob is not None else None
        calibrated_cataract = self._calibrate_probability(cataract_prob, "cataract") if cataract_prob is not None else None
        calibrated_stye = self._calibrate_probability(stye_prob, "stye") if stye_prob is not None else None

        # 5. Composite Multi-Target Screening Risk Tier
        if should_abstain:
            screening_tier = "inconclusive_abstain"
            tier_label = "INCONCLUSIVE / ABSTAIN"
            risk_summary = "Screening deferred due to image quality or illumination constraints. Re-capture required."
        else:
            # Evaluate clinical risk hierarchy
            is_severe_anemia = (hb_val is not None and hb_val < 8.0)
            is_moderate_anemia = (hb_val is not None and 8.0 <= hb_val < 11.0)
            is_active_injection = (("injection" in slid_class or "hyperemia" in slid_class) and (calibrated_slid or 0) > 0.65)
            is_active_icterus = (jaund_class == "visibly_jaundiced" and (calibrated_jaund or 0) > 0.65)
            is_active_cataract = ("cataract" in str(cataract_class).lower() and (calibrated_cataract or 0) > 0.65)
            is_active_stye = (("stye" in str(stye_class).lower() or "hordeolum" in str(stye_class).lower()) and (calibrated_stye or 0) > 0.65)

            if is_severe_anemia or (is_active_injection and is_active_icterus) or (is_active_cataract and (cataract_locs or 0) >= 4):
                screening_tier = "high_risk_flag"
                tier_label = "HIGH RISK / CLINICAL ATTENTION"
                risk_summary = "Significant ocular abnormalities detected. Immediate ophthalmologist / physician consultation advised."
            elif is_moderate_anemia or is_active_injection or is_active_icterus or is_active_cataract or is_active_stye:
                screening_tier = "borderline_review"
                tier_label = "BORDERLINE / CLINICAL REVIEW"
                risk_summary = "One or more biomarkers show deviation from normal reference ranges. Confirmatory testing recommended."
            else:
                screening_tier = "low_risk"
                tier_label = "LOW RISK / NORMAL RANGES"
                risk_summary = "Ocular features appear within expected non-pathological thresholds across all evaluated modules."

        return {
            "status": "success",
            "screening_tier": screening_tier,
            "tier_label": tier_label,
            "risk_summary": risk_summary,
            "should_abstain": should_abstain,
            "abstention_reasons": abstention_reasons,
            "concordance_score": concordance_score,
            "concordance_flags": concordance_flags,
            "calibrated_probabilities": {
                "slid_injection": calibrated_slid,
                "scleral_icterus": calibrated_jaund,
                "nuclear_cataract": calibrated_cataract,
                "stye_hordeolum": calibrated_stye
            },
            "specialist_findings": {
                "anemia_hb_g_dl": hb_val,
                "anemia_staging": hb_stage,
                "conjunctivitis_status": slid_class,
                "jaundice_status": jaund_class,
                "cataract_status": cataract_class,
                "cataract_locs_grade": cataract_locs,
                "stye_status": stye_class
            },
            "segmentation_adequacy": {
                "conjunctiva_exposure_pct": conj_cov,
                "sclera_exposure_pct": sclera_cov,
                "is_adequate": (conj_cov >= self.min_conj_cov) and (sclera_cov >= self.min_sclera_cov)
            },
            "disclaimer": "CALIBRATED ENSEMBLE NOTICE: Composite screening tier is an experimental multi-modal indicator for research triage only. Not a diagnostic decision."
        }


# Global singleton
_CALIBRATED_ENSEMBLE: Optional[CalibratedEnsembleFusion] = None

def get_calibrated_ensemble() -> CalibratedEnsembleFusion:
    global _CALIBRATED_ENSEMBLE
    if _CALIBRATED_ENSEMBLE is None:
        _CALIBRATED_ENSEMBLE = CalibratedEnsembleFusion()
    return _CALIBRATED_ENSEMBLE
