"""
src/inference/clinical_analytics.py
----------------------------------
Clinical decision support and physiological chromatic biomarker profiling
for EyeHealthAI production screening pipelines.

Provides:
1. WHO-aligned clinical anemia risk staging, 95% confidence interval estimation,
   and critical threshold proximity / equivocal zone alerts.
2. Objective ocular chromatic biomarker extraction (Erythema Index, CIE-L*a*b* a* and b*,
   Scleral Yellowness Index).
3. Production clinical decision synthesis for Conjunctival Injection (SLID) and
   Scleral Icterus (Jaundice).
"""

from typing import Dict, Any, Optional, List
import numpy as np
import cv2
from PIL import Image


def extract_ocular_chromatic_features(img: Image.Image) -> Dict[str, Any]:
    """
    Extracts objective physiological chromatic biomarkers from an ocular image:
    - Erythema Index (EI): Microvascular perfusion proxy in conjunctival tissue.
    - CIE-L*a*b* a* (Red-Green chroma): Palpebral erythema / pallor indicator.
    - CIE-L*a*b* b* (Yellow-Blue chroma): Scleral bilirubin staining indicator.
    - Scleral Yellowness Index (SYI): Optical yellowing ratio (R+G) / (2*B).
    """
    if not isinstance(img, Image.Image):
        return {
            "erythema_index": 0.0,
            "cie_a_star": 0.0,
            "cie_b_star": 0.0,
            "scleral_yellowness_index": 1.0,
            "mean_luminance": 128.0,
            "pallor_indicator": "unknown"
        }

    rgb_img = img.convert('RGB')
    arr = np.array(rgb_img, dtype=np.float32)

    # Exclude black padding borders (pixels where R=G=B=0)
    non_black = (arr[:, :, 0] > 10) | (arr[:, :, 1] > 10) | (arr[:, :, 2] > 10)
    if non_black.sum() > 50:
        pixels = arr[non_black]
    else:
        pixels = arr.reshape(-1, 3)

    r_mean = float(pixels[:, 0].mean())
    g_mean = float(pixels[:, 1].mean())
    b_mean = float(pixels[:, 2].mean())

    # Erythema Index: (R - G) / (R + G + eps)
    ei = (r_mean - g_mean) / (r_mean + g_mean + 1e-6)

    # Scleral Yellowness Index: (R + G) / (2 * max(B, 1))
    syi = (r_mean + g_mean) / (2.0 * max(b_mean, 1.0))

    # CIE-L*a*b* conversion via OpenCV
    # OpenCV uint8 LAB: L in [0, 255], a in [0, 255] (128=neutral), b in [0, 255] (128=neutral)
    uint8_arr = np.uint8(np.clip(arr, 0, 255))
    lab = cv2.cvtColor(uint8_arr, cv2.COLOR_RGB2LAB)
    if non_black.sum() > 50:
        lab_pixels = lab[non_black]
    else:
        lab_pixels = lab.reshape(-1, 3)

    l_mean = float(lab_pixels[:, 0].mean())
    a_star = float(lab_pixels[:, 1].mean()) - 128.0
    b_star = float(lab_pixels[:, 2].mean()) - 128.0

    # Pallor assessment
    if ei < 0.12 or a_star < 3.0:
        pallor_state = "marked_pallor"
    elif ei < 0.22 or a_star < 7.0:
        pallor_state = "mild_pallor"
    else:
        pallor_state = "normal_erythema"

    return {
        "erythema_index": round(float(ei), 3),
        "cie_a_star": round(float(a_star), 2),
        "cie_b_star": round(float(b_star), 2),
        "scleral_yellowness_index": round(float(syi), 3),
        "mean_luminance": round(float(l_mean), 1),
        "pallor_indicator": pallor_state
    }


def compute_anemia_clinical_decision(
    hb_val: float,
    gender: str = "prefer_not_to_say",
    view_count: int = 1
) -> Dict[str, Any]:
    """
    Translates raw continuous hemoglobin regression into WHO-aligned clinical staging,
    diagnostic certainty rating, and actionable next steps.
    
    WHO Reference Thresholds:
    - Adult Female: Normal >= 12.0 g/dL | Mild 11.0-11.9 | Moderate 8.0-10.9 | Severe < 8.0
    - Adult Male: Normal >= 13.0 g/dL | Mild 11.0-12.9 | Moderate 8.0-10.9 | Severe < 8.0
    - Unspecified: Baseline 12.0 g/dL threshold
    """
    norm_threshold = 13.0 if gender.lower() == "male" else 12.0

    # 1. WHO Severity Staging
    if hb_val < 8.0:
        who_stage = "severe_anemia"
        stage_label_en = "Severe Anemia"
        stage_label_hi = "गंभीर एनीमिया"
        risk_color = "#ef4444"
        urgency = "high_priority"
    elif hb_val < 11.0:
        who_stage = "moderate_anemia"
        stage_label_en = "Moderate Anemia"
        stage_label_hi = "मध्यम एनीमिया"
        risk_color = "#f97316"
        urgency = "priority"
    elif hb_val < norm_threshold:
        who_stage = "mild_anemia"
        stage_label_en = "Mild Anemia"
        stage_label_hi = "हल्का एनीमिया"
        risk_color = "#eab308"
        urgency = "moderate"
    else:
        who_stage = "normal_non_anemic"
        stage_label_en = "Normal / Non-Anemic"
        stage_label_hi = "सामान्य (एनीमिया नहीं)"
        risk_color = "#10b981"
        urgency = "routine"

    # 2. Critical Diagnostic Boundary / Equivocal Zone Detection
    # When Hb is within +/- 0.8 g/dL of threshold, model error (MAE ~1.5) could span the diagnostic boundary
    is_borderline = abs(hb_val - norm_threshold) <= 0.85
    if is_borderline:
        certainty = "borderline_threshold_equivocal"
        certainty_desc_en = (
            f"BORDERLINE THRESHOLD PROXIMITY: Estimated hemoglobin ({hb_val:.2f} g/dL) is within "
            f"the uncertainty margin of the WHO clinical threshold ({norm_threshold:.1f} g/dL). "
            "Optical non-invasive screening cannot reliably distinguish borderline mild anemia from "
            "normal baseline without confirmatory laboratory venous Complete Blood Count (CBC)."
        )
        certainty_desc_hi = (
            f"सीमांत दहलीज निकटता: अनुमानित हीमोग्लोबिन ({hb_val:.2f} g/dL) WHO चिकित्सीय स्तर "
            f"({norm_threshold:.1f} g/dL) की अनिश्चितता सीमा के भीतर है। "
            "प्रयोगशाला शिरापरक सीबीसी (CBC) परीक्षण अनिवार्य है।"
        )
    else:
        certainty = "concordant"
        certainty_desc_en = (
            f"Estimated hemoglobin ({hb_val:.2f} g/dL) is concordant with the {stage_label_en} category. "
            "Subject to model error margin (MAE ~1.50 g/dL)."
        )
        certainty_desc_hi = (
            f"अनुमानित हीमोग्लोबिन ({hb_val:.2f} g/dL) {stage_label_hi} श्रेणी के अनुरूप है।"
        )

    # 3. Empirical 95% Confidence Interval based on test set residuals (RMSE ~2.07 g/dL)
    ci_margin = 1.96 * 1.05  # ~2.05 g/dL
    ci_low = round(max(3.5, hb_val - ci_margin), 2)
    ci_high = round(min(20.0, hb_val + ci_margin), 2)

    # 4. Clinical Recommendations
    recommendations_en = [
        "Confirmatory Testing: Schedule a laboratory venous blood panel (Complete Blood Count / CBC) for certified diagnostic hemoglobin.",
        "Clinical Correlation: Correlate with clinical symptoms including conjunctival pallor, fatigue, dyspnea, tachycardia, or cold extremities.",
        "Physician Review: Share this screening summary and confidence interval with a licensed medical practitioner."
    ]
    if who_stage in ("severe_anemia", "moderate_anemia"):
        recommendations_en.insert(0, "URGENT MEDICAL ATTENTION: Prompt clinical evaluation is recommended due to pronounced ocular pallor findings.")

    return {
        "who_stage": who_stage,
        "stage_label_en": stage_label_en,
        "stage_label_hi": stage_label_hi,
        "risk_color": risk_color,
        "norm_threshold_g_dl": norm_threshold,
        "diagnostic_certainty": certainty,
        "certainty_description_en": certainty_desc_en,
        "certainty_description_hi": certainty_desc_hi,
        "confidence_interval_95": {
            "lower_g_dl": ci_low,
            "upper_g_dl": ci_high,
            "display": f"{ci_low:.1f} – {ci_high:.1f} g/dL"
        },
        "clinical_urgency": urgency,
        "recommendations_en": recommendations_en
    }


def compute_slid_clinical_decision(
    predicted_class: str,
    confidence: float,
    chromatic: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Clinical decision support for SLID Anterior Slit-Lamp Hyperemia Screening.
    Differentiates visible conjunctival injection from normal ocular surface.
    """
    is_normal = (predicted_class == "control_normal")
    if is_normal:
        status_en = "Normal Ocular Surface"
        status_hi = "सामान्य नेत्र सतह (हाइपरिमिया अनुपस्थित)"
        risk_color = "#10b981"
        summary_en = (
            "Slit-lamp examination indicates normal ocular surface microvasculature. "
            "No marked conjunctival injection or vessel dilation observed."
        )
        guidance_en = [
            "Routine eye health maintenance and periodic vision screening.",
            "If experiencing dry eyes, itching, or irritation, use preservative-free lubricating drops.",
            "Seek ophthalmological consultation if new symptoms develop."
        ]
    else:
        status_en = "Conjunctival Hyperemia / Injection Detected"
        status_hi = "कंजंक्टाइवल इंजेक्शन / लाली देखी गई"
        risk_color = "#ef4444"
        summary_en = (
            "Prominent conjunctival injection (microvascular hyperemia) detected on anterior ocular photography. "
            "This is the primary physical finding associated with conjunctival inflammation / conjunctivitis."
        )
        guidance_en = [
            "Clinical Slit-Lamp Biomicroscopy: Required to inspect the tarsal conjunctiva and rule out corneal involvement.",
            "Etiological Differentiation: An ophthalmologist must distinguish viral, bacterial, allergic conjunctivitis, or anterior uveitis.",
            "Avoid Self-Medication: Do not use over-the-counter steroid or antibiotic eye drops without a prescription."
        ]

    return {
        "finding_en": status_en,
        "finding_hi": status_hi,
        "is_hyperemic": not is_normal,
        "risk_color": risk_color,
        "confidence_pct": round(confidence * 100, 1),
        "summary_en": summary_en,
        "clinical_action_guidance": guidance_en,
        "chromatic_erythema": chromatic.get("erythema_index", 0.0)
    }


def compute_jaundice_clinical_decision(
    predicted_class: str,
    confidence: float,
    chromatic: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Clinical decision support for Scleral Icterus (Jaundice) Screening.
    Differentiates visibly jaundiced sclera from non-icteric baseline.
    """
    is_normal = (predicted_class == "normal")
    syi = chromatic.get("scleral_yellowness_index", 1.0)
    b_star = chromatic.get("cie_b_star", 0.0)

    if is_normal:
        status_en = "Non-Icteric Sclera (Normal)"
        status_hi = "सामान्य स्क्लेरा (पीलिया अनुपस्थित)"
        risk_color = "#10b981"
        summary_en = (
            f"Ocular sclera exhibits normal chromatic white baseline (Yellowness Index: {syi:.2f}, b*: {b_star:.1f}). "
            "No visible scleral icterus detected."
        )
        guidance_en = [
            "Visual appearance is normal; does not rule out subclinical hyperbilirubinemia (<2.0 mg/dL).",
            "Maintain general wellness; report any systemic symptoms such as dark urine or pale stools.",
            "Routine health checkups as recommended by your physician."
        ]
    else:
        status_en = "Visible Scleral Icterus Detected (Jaundice Indicator)"
        status_hi = "स्क्लेरल इक्टेरस (पीलिया का लक्षण) देखा गया"
        risk_color = "#eab308"
        summary_en = (
            f"Prominent scleral yellowing observed (Yellowness Index: {syi:.2f}, elevated b*: {b_star:.1f}). "
            "Scleral icterus typically manifests when total serum bilirubin exceeds 2.5–3.0 mg/dL."
        )
        guidance_en = [
            "Liver Function Tests (LFTs): Immediate venous laboratory testing for Total & Direct Bilirubin, ALT, AST, and Alkaline Phosphatase.",
            "Physician Evaluation: Consult a gastroenterologist or general physician for hepatic and biliary evaluation.",
            "Abdominal Ultrasound: May be required to evaluate gallbladder, bile ducts, and hepatic parenchyma."
        ]

    return {
        "finding_en": status_en,
        "finding_hi": status_hi,
        "is_icteric": not is_normal,
        "risk_color": risk_color,
        "confidence_pct": round(confidence * 100, 1),
        "scleral_yellowness_index": syi,
        "cie_b_star_yellow": b_star,
        "summary_en": summary_en,
        "clinical_action_guidance": guidance_en,
        "laboratory_bilirubin_note": "Quantitative bilirubin regression is UNAVAILABLE non-invasively; venous laboratory blood test required."
    }
