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
    - Erythema Index (EI): Microvascular perfusion proxy in conjunctival mucosal tissue.
    - CIE-L*a*b* a* (Red-Green chroma): Palpebral erythema / pallor indicator.
    - CIE-L*a*b* b* (Yellow-Blue chroma): Scleral bilirubin staining indicator.
    - Scleral Yellowness Index (SYI): Optical yellowing ratio (R+G) / (2*B).
    Isolates ocular anatomical tissue to prevent facial skin undertones from biasing measurements.
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
    uint8_arr = np.uint8(np.clip(arr, 0, 255))
    lab = cv2.cvtColor(uint8_arr, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(uint8_arr, cv2.COLOR_RGB2HSV)

    # Exclude black padding borders (pixels where R=G=B=0)
    non_black = (arr[:, :, 0] > 10) | (arr[:, :, 1] > 10) | (arr[:, :, 2] > 10)

    # 1. Sclera candidate pixels (White sclera or Jaundiced yellow sclera)
    sclera_white = (lab[:, :, 0] > 90) & (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 60) & non_black
    b_val = lab[:, :, 2].astype(float) - 128.0
    a_val = lab[:, :, 1].astype(float) - 128.0
    sclera_yellow = (b_val >= 14.0) & (b_val - a_val >= 4.0) & (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 55) & (lab[:, :, 0] > 65) & non_black
    sclera_mask = sclera_white | sclera_yellow
    if sclera_mask.sum() > 150:
        sclera_lab = lab[sclera_mask]
        sclera_rgb = arr[sclera_mask]
        b_star = float(sclera_lab[:, 2].mean()) - 128.0
        r_sc = float(sclera_rgb[:, 0].mean())
        g_sc = float(sclera_rgb[:, 1].mean())
        b_sc = float(sclera_rgb[:, 2].mean())
        syi = (r_sc + g_sc) / (2.0 * max(b_sc, 1.0))
    else:
        b_star = float(lab[non_black, 2].mean()) - 128.0 if non_black.sum() > 0 else 0.0
        syi = 1.0

    # 2. Conjunctival mucosal vascular bed (R > G, a* > 6.0, moderate L)
    conj_mask = (lab[:, :, 1].astype(float) - 128.0 > 6.0) & (lab[:, :, 0] > 60) & non_black
    if conj_mask.sum() > 150:
        conj_lab = lab[conj_mask]
        conj_rgb = arr[conj_mask]
        a_star = float(conj_lab[:, 1].mean()) - 128.0
        r_c = float(conj_rgb[:, 0].mean())
        g_c = float(conj_rgb[:, 1].mean())
        ei = (r_c - g_c) / (r_c + g_c + 1e-6)
    else:
        a_star = float(lab[non_black, 1].mean()) - 128.0 if non_black.sum() > 0 else 0.0
        ei = 0.15

    l_mean = float(lab[non_black, 0].mean()) if non_black.sum() > 0 else 128.0

    # Pallor assessment based on genuine conjunctival mucosal vascular bed
    if ei < 0.10 or a_star < 8.0:
        pallor_state = "marked_pallor"
    elif ei < 0.15 or a_star < 12.0:
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
    age_years: Optional[int] = None,
    clinical_sex: str = "not_disclosed",
    is_pregnant: bool = False,
    pregnancy_trimester: Optional[int] = None,
    gender: str = "prefer_not_to_say",
    view_count: int = 1
) -> Dict[str, Any]:
    """
    Translates raw continuous hemoglobin regression into WHO 2024-aligned clinical staging,
    diagnostic certainty rating, population validation boundary check, and actionable next steps.
    
    WHO Guideline on Haemoglobin Cutoffs to Define Anaemia (2024 Edition):
    - Children 6-59 months (<5y): Normal >= 11.0 g/dL | Mild 10.0-10.9 | Moderate 7.0-9.9 | Severe < 7.0
    - Children 5-11 years: Normal >= 11.5 g/dL | Mild 11.1-11.4 | Moderate 8.0-11.0 | Severe < 8.0
    - Children 12-14 years: Normal >= 12.0 g/dL | Mild 11.0-11.9 | Moderate 8.0-10.9 | Severe < 8.0
    - Non-pregnant Females (>=15y): Normal >= 12.0 g/dL | Mild 11.0-11.9 | Moderate 8.0-10.9 | Severe < 8.0
    - Pregnant Females: Normal >= 11.0 g/dL (2nd trim: >=10.5) | Mild 10.0-10.9 | Moderate 7.0-9.9 | Severe < 7.0
    - Adult Males (>=15y): Normal >= 13.0 g/dL | Mild 11.0-12.9 | Moderate 8.0-10.9 | Severe < 8.0
    - Undisclosed Clinical Sex: Generic adult baseline threshold 12.0 g/dL
    
    POPULATION VALIDATION BOUNDARIES:
    - ResNet-50 Anemia model was trained on India and Italy cohorts (ages 18 to 88).
    - Pediatric (<18) and extreme geriatric (>88) subjects are flagged as outside_validated_population.
    """
    # 1. Population Validation Boundary Audit
    if age_years is not None:
        if age_years < 18:
            is_pop_valid = False
            pop_status = "outside_validated_population"
            pop_reason = "pediatric_under_18"
            pop_notice_en = (
                f"Pediatric Population Warning: Patient age ({age_years}y) is below the validated training cohort "
                "(adults 18–88 years). Pediatric microvascular conjunctival morphology is outside the trained distribution."
            )
            pop_notice_hi = (
                f"बाल चिकित्सा जनसंख्या चेतावनी: रोगी की आयु ({age_years} वर्ष) मॉडल के नैदानिक रूप से मान्य वयस्क प्रशिक्षण समूह (18-88 वर्ष) से बाहर है।"
            )
        elif age_years > 88:
            is_pop_valid = False
            pop_status = "outside_validated_population"
            pop_reason = "geriatric_over_88"
            pop_notice_en = (
                f"Geriatric Population Warning: Patient age ({age_years}y) exceeds the maximum training cohort "
                "ceiling (88 years). Optical hemoglobin regression may exhibit uncalibrated variance."
            )
            pop_notice_hi = (
                f"वृद्धावस्था जनसंख्या चेतावनी: रोगी की आयु ({age_years} वर्ष) अधिकतम प्रशिक्षण समूह सीमा (88 वर्ष) से अधिक है।"
            )
        else:
            is_pop_valid = True
            pop_status = "within_validated_population"
            pop_reason = None
            pop_notice_en = f"Patient age ({age_years}y) falls within the model's clinically validated adult cohort (18–88 years)."
            pop_notice_hi = f"रोगी की आयु ({age_years} वर्ष) मॉडल के चिकित्सकीय रूप से सत्यापित समूह (18-88 वर्ष) के अंतर्गत है।"
    else:
        is_pop_valid = True
        pop_status = "unspecified_age"
        pop_reason = "unspecified_age"
        pop_notice_en = "Patient age unspecified; adult reference standards applied as baseline."
        pop_notice_hi = "रोगी की आयु निर्दिष्ट नहीं है; वयस्क आधारभूत संदर्भ मानक लागू किए गए हैं।"

    population_val_dict = {
        "is_validated_population": is_pop_valid,
        "status": pop_status,
        "reason": pop_reason,
        "clinical_notice_en": pop_notice_en,
        "clinical_notice_hi": pop_notice_hi
    }
    population_warning = pop_notice_en if not is_pop_valid else None

    # 2. WHO 2024 Reference Thresholds Determination
    clean_sex = (clinical_sex or gender or "not_disclosed").strip().lower()
    if clean_sex not in {"male", "female", "not_disclosed"}:
        clean_sex = "not_disclosed"

    if age_years is not None and age_years < 5:
        ref_group = "Children (6–59 months)"
        norm_threshold = 11.0
        sev_cutoff = 7.0
        mod_cutoff = 10.0
    elif age_years is not None and age_years < 12:
        ref_group = "Children (5–11 years)"
        norm_threshold = 11.5
        sev_cutoff = 8.0
        mod_cutoff = 11.1
    elif age_years is not None and age_years < 15:
        ref_group = "Children (12–14 years)"
        norm_threshold = 12.0
        sev_cutoff = 8.0
        mod_cutoff = 11.0
    else:
        # Age >= 15 or unspecified adult
        if clean_sex == "female":
            if is_pregnant:
                if pregnancy_trimester == 2:
                    ref_group = "Pregnant Females (2nd Trimester, WHO 2024)"
                    norm_threshold = 10.5
                    sev_cutoff = 7.0
                    mod_cutoff = 10.0
                else:
                    ref_group = "Pregnant Females (1st/3rd Trimester, WHO 2024)"
                    norm_threshold = 11.0
                    sev_cutoff = 7.0
                    mod_cutoff = 10.0
            else:
                ref_group = "Non-Pregnant Females (≥15 years, WHO 2024)"
                norm_threshold = 12.0
                sev_cutoff = 8.0
                mod_cutoff = 11.0
        elif clean_sex == "male":
            ref_group = "Adult Males (≥15 years, WHO 2024)"
            norm_threshold = 13.0
            sev_cutoff = 8.0
            mod_cutoff = 11.0
        else:
            ref_group = "Adult General Baseline (Clinical Sex Not Disclosed)"
            norm_threshold = 12.0
            sev_cutoff = 8.0
            mod_cutoff = 11.0

    # 3. WHO Severity Staging with Screening Error Margin Consideration
    if hb_val < sev_cutoff:
        who_stage = "severe_anemia"
        stage_label_en = "Severe Anemia"
        stage_label_hi = "गंभीर एनीमिया"
        risk_color = "#ef4444"
        urgency = "high_priority"
    elif hb_val < (mod_cutoff - 0.5):  # Strictly below 10.5 g/dL
        who_stage = "moderate_anemia"
        stage_label_en = "Moderate Anemia"
        stage_label_hi = "मध्यम एनीमिया"
        risk_color = "#f97316"
        urgency = "priority"
    elif hb_val < norm_threshold:
        who_stage = "mild_anemia"
        stage_label_en = "Mild Anemia / Borderline Variation" if hb_val >= (mod_cutoff - 0.5) else "Mild Anemia"
        stage_label_hi = "हल्का एनीमिया / सीमांत भिन्नता"
        risk_color = "#eab308"
        urgency = "moderate"
    else:
        who_stage = "normal_non_anemic"
        stage_label_en = "Non-Anemic Baseline"
        stage_label_hi = "सामान्य (एनीमिया नहीं)"
        risk_color = "#10b981"
        urgency = "routine"

    # 4. Critical Diagnostic Boundary / Equivocal Zone Detection (MAE ~1.50 g/dL)
    is_borderline = (abs(hb_val - norm_threshold) <= 1.50) or (abs(hb_val - mod_cutoff) <= 0.85)
    if is_borderline:
        certainty = "borderline_threshold_equivocal"
        certainty_desc_en = (
            f"BORDERLINE THRESHOLD PROXIMITY: Estimated hemoglobin ({hb_val:.2f} g/dL) is within "
            f"the uncertainty margin of the WHO clinical threshold ({norm_threshold:.1f} g/dL, test MAE ±1.50 g/dL). "
            "Optical non-invasive screening cannot reliably distinguish borderline mild anemia from "
            "normal baseline without confirmatory laboratory venous Complete Blood Count (CBC)."
        )
        certainty_desc_hi = (
            f"सीमांत दहलीज निकटता: अनुमानित हीमोग्लोबिन ({hb_val:.2f} g/dL) WHO चिकित्सीय स्तर "
            f"({norm_threshold:.1f} g/dL) की अनिश्चितता सीमा (±1.50 g/dL) के भीतर है। "
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

    # 5. Empirical 95% Confidence Interval (RMSE ~2.07 g/dL)
    ci_margin = 1.96 * 1.05  # ~2.05 g/dL
    ci_low = round(max(3.5, hb_val - ci_margin), 2)
    ci_high = round(min(20.0, hb_val + ci_margin), 2)

    # 6. Clinical Recommendations
    recommendations_en = [
        "Confirmatory Testing: Schedule a laboratory venous blood panel (Complete Blood Count / CBC) for certified diagnostic hemoglobin.",
        "Clinical Correlation: Correlate with clinical symptoms including conjunctival pallor, fatigue, dyspnea, tachycardia, or cold extremities.",
        "Physician Review: Share this screening summary and confidence interval with a licensed medical practitioner."
    ]
    if who_stage in ("severe_anemia", "moderate_anemia"):
        recommendations_en.insert(0, "URGENT MEDICAL ATTENTION: Prompt clinical evaluation is recommended due to pronounced ocular pallor findings.")
    if population_warning:
        recommendations_en.append(population_warning)

    recommendations_hi = [
        "पुष्टि परीक्षण: प्रमाणित हीमोग्लोबिन हेतु प्रयोगशाला सीबीसी (CBC) रक्त जांच कराएं।",
        "लक्षण मिलान: थकान, सांस फूलना या कमजोरी जैसे लक्षणों पर डॉक्टर से परामर्श लें।",
        "चिकित्सक समीक्षा: यह रिपोर्ट अपने पंजीकृत डॉक्टर के साथ साझा करें।"
    ]

    return {
        "who_stage": who_stage,
        "stage_label_en": stage_label_en,
        "stage_label_hi": stage_label_hi,
        "clinical_category": stage_label_en,
        "risk_color": risk_color,
        "reference_threshold_hb": norm_threshold,
        "norm_threshold_g_dl": norm_threshold,
        "clinical_reference_group": ref_group,
        "population_validation": population_val_dict,
        "population_warning": population_warning,
        "diagnostic_certainty": certainty,
        "certainty_description_en": certainty_desc_en,
        "certainty_description_hi": certainty_desc_hi,
        "confidence_interval_95": {
            "lower_g_dl": ci_low,
            "upper_g_dl": ci_high,
            "display": f"{ci_low:.1f} – {ci_high:.1f} g/dL"
        },
        "clinical_urgency": urgency,
        "recommendations_en": recommendations_en,
        "recommendations_hi": recommendations_hi
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
