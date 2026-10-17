"""
src/inference/personalized_ai.py
---------------------------------
Profile-Aware OpenAI Vision-Assisted Personalized Clinical Explanation Service.

Provides individualized, research-grade multimodal explanations integrating:
  1. Cropped ocular screening photograph
  2. Patient demographic profile (age_years, clinical_sex, pregnancy status)
  3. Quantitative model predictions (Hb regression, SLID classification, Scleral Icterus)
  4. Anatomical tissue coverage and quality metrics
  5. WHO 2024 age/sex-specific reference ranges and population boundary validation

MANDATORY DATA MINIMIZATION & ETHICAL GUARDRAILS:
  - Requires explicit participant informed consent (`external_ai_consent: true`).
  - STRICT PII FILTER: Patient names, email addresses, phone numbers, full dates of birth,
    and raw database IDs are NEVER transmitted.
  - Demographics are strictly used post-hoc to contextualize clinical reference ranges;
    they are NEVER used to distort computer vision feature representations.
  - If external consent is withheld or OpenAI is unreachable/unconfigured, degrades gracefully
    to an authoritative local deterministic bilingual clinical explanation.
"""

import os
import io
import time
import json
import base64
import logging
from typing import Optional, Dict, Any, List
from PIL import Image

from src.config import settings

logger = logging.getLogger(__name__)


class PersonalizedAIService:
    """
    Multimodal clinical research explanation service using OpenAI Vision
    with deterministic fallback and strict PII protection.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", None)
        self.model_name = (
            model_name
            or os.getenv("OPENAI_VISION_MODEL")
            or getattr(settings, "OPENAI_VISION_MODEL", None)
            or os.getenv("OPENAI_MODEL_NAME")
            or "gpt-4o-mini"
        )
        self._openai_client = None

    def _get_client(self):
        if self._openai_client is None and self.api_key and len(self.api_key.strip()) > 5:
            try:
                import openai
                self._openai_client = openai.OpenAI(api_key=self.api_key.strip(), timeout=25.0)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client for personalized AI: {e}")
                self._openai_client = None
        return self._openai_client

    def generate_personalized_explanation(
        self,
        image: Optional[Image.Image],
        age_years: Optional[int],
        clinical_sex: str = "not_disclosed",
        is_pregnant: bool = False,
        pregnancy_trimester: Optional[int] = None,
        anemia_result: Optional[Dict[str, Any]] = None,
        slid_result: Optional[Dict[str, Any]] = None,
        jaundice_result: Optional[Dict[str, Any]] = None,
        quality_result: Optional[Dict[str, Any]] = None,
        segmentation_result: Optional[Dict[str, Any]] = None,
        clinical_decision: Optional[Dict[str, Any]] = None,
        external_ai_consent: bool = False,
        preferred_language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generates a profile-aware personalized analysis.
        If external_ai_consent is True and OpenAI is available, uses OpenAI Vision.
        Otherwise, uses the local deterministic clinical explanation engine.
        """
        # Ensure age_years is integer if present
        if age_years is not None:
            try:
                age_years = int(age_years)
            except (ValueError, TypeError):
                age_years = None

        clinical_sex = (clinical_sex or "not_disclosed").strip().lower()

        # Step 1: Check if external AI consent was granted
        if not external_ai_consent:
            logger.info("External AI consent withheld. Generating local deterministic explanation.")
            return self._generate_deterministic_explanation(
                age_years=age_years,
                clinical_sex=clinical_sex,
                is_pregnant=is_pregnant,
                pregnancy_trimester=pregnancy_trimester,
                anemia_result=anemia_result,
                slid_result=slid_result,
                jaundice_result=jaundice_result,
                quality_result=quality_result,
                segmentation_result=segmentation_result,
                clinical_decision=clinical_decision,
                reason="consent_withheld"
            )

        client = self._get_client()
        if client is None:
            logger.info("OpenAI client not configured or key missing. Using local deterministic explanation.")
            return self._generate_deterministic_explanation(
                age_years=age_years,
                clinical_sex=clinical_sex,
                is_pregnant=is_pregnant,
                pregnancy_trimester=pregnancy_trimester,
                anemia_result=anemia_result,
                slid_result=slid_result,
                jaundice_result=jaundice_result,
                quality_result=quality_result,
                segmentation_result=segmentation_result,
                clinical_decision=clinical_decision,
                reason="openai_unconfigured"
            )

        # Step 2: Prepare sanitized, minimized payload
        sanitized_summary = self._build_sanitized_payload(
            age_years=age_years,
            clinical_sex=clinical_sex,
            is_pregnant=is_pregnant,
            pregnancy_trimester=pregnancy_trimester,
            anemia_result=anemia_result,
            slid_result=slid_result,
            jaundice_result=jaundice_result,
            quality_result=quality_result,
            segmentation_result=segmentation_result,
            clinical_decision=clinical_decision
        )

        # Encode image safely (max 512px on longest side for bandwidth & minimization)
        b64_img = None
        if image is not None:
            try:
                img_copy = image.copy()
                img_copy.thumbnail((512, 512), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img_copy.save(buf, format="JPEG", quality=85)
                b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to encode image for personalized AI: {e}")

        # Step 3: Invoke OpenAI Vision API
        t0 = time.time()
        try:
            explanation = self._call_openai_vision(
                client=client,
                sanitized_payload=sanitized_summary,
                b64_image=b64_img
            )
            latency_ms = round((time.time() - t0) * 1000, 1)
            explanation["latency_ms"] = latency_ms
            explanation["status"] = "completed"
            explanation["provider"] = "openai_vision"
            explanation["model_used"] = self.model_name
            explanation["external_ai_consent"] = True
            return explanation

        except Exception as e:
            logger.warning(f"OpenAI personalized vision analysis failed ({e}). Falling back to deterministic engine.")
            fallback = self._generate_deterministic_explanation(
                age_years=age_years,
                clinical_sex=clinical_sex,
                is_pregnant=is_pregnant,
                pregnancy_trimester=pregnancy_trimester,
                anemia_result=anemia_result,
                slid_result=slid_result,
                jaundice_result=jaundice_result,
                quality_result=quality_result,
                segmentation_result=segmentation_result,
                clinical_decision=clinical_decision,
                reason=f"api_error: {str(e)[:100]}"
            )
            fallback["latency_ms"] = round((time.time() - t0) * 1000, 1)
            return fallback

    def _build_sanitized_payload(
        self,
        age_years: Optional[int],
        clinical_sex: str,
        is_pregnant: bool,
        pregnancy_trimester: Optional[int],
        anemia_result: Optional[Dict[str, Any]],
        slid_result: Optional[Dict[str, Any]],
        jaundice_result: Optional[Dict[str, Any]],
        quality_result: Optional[Dict[str, Any]],
        segmentation_result: Optional[Dict[str, Any]],
        clinical_decision: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extracts strictly clinical, non-PII features for LLM evaluation.
        Guarantees no patient name, phone, email, address, or full DOB.
        """
        # Anemia details
        anemia_dict = {}
        if anemia_result:
            hb = anemia_result.get("predicted_hb_g_dl") or anemia_result.get("view_mean")
            if hb is None and anemia_result.get("prediction"):
                preds = anemia_result.get("prediction")
                if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
                    hb = preds[0].get("hb_pred_g_dl") or preds[0].get("predicted_hb_g_dl")
            ci_95 = anemia_result.get("confidence_interval_95")
            if not ci_95 and clinical_decision:
                ci_95 = clinical_decision.get("confidence_interval_95")
            anemia_dict = {
                "estimated_hb_g_dl": round(float(hb), 2) if hb is not None else None,
                "model_status": anemia_result.get("status", "unknown"),
                "confidence_interval_95": ci_95,
                "evaluation_status": anemia_result.get("evaluation_status", "executed")
            }

        # Decision details
        decision_dict = {}
        if clinical_decision:
            decision_dict = {
                "clinical_reference_group": clinical_decision.get("clinical_reference_group"),
                "reference_threshold_hb": clinical_decision.get("reference_threshold_hb"),
                "who_guideline": clinical_decision.get("who_guideline"),
                "clinical_category": clinical_decision.get("clinical_category"),
                "borderline_equivocal": clinical_decision.get("borderline_equivocal", False),
                "population_validation": clinical_decision.get("population_validation", {})
            }

        # SLID / Conjunctivitis details
        slid_dict = {}
        if slid_result:
            slid_pred = slid_result.get("prediction")
            slid_first = (slid_pred[0] if isinstance(slid_pred, list) and len(slid_pred) > 0 and isinstance(slid_pred[0], dict)
                          else (slid_pred if isinstance(slid_pred, dict) else {}))
            slid_dict = {
                "status": slid_result.get("status"),
                "prediction": slid_first.get("predicted_class", slid_pred),
                "probability": slid_first.get("confidence", slid_result.get("probability"))
            }

        # Jaundice details
        jaundice_dict = {}
        if jaundice_result:
            jaund_pred = jaundice_result.get("prediction")
            jaund_first = (jaund_pred[0] if isinstance(jaund_pred, list) and len(jaund_pred) > 0 and isinstance(jaund_pred[0], dict)
                           else (jaund_pred if isinstance(jaund_pred, dict) else {}))
            jaundice_dict = {
                "status": jaundice_result.get("status"),
                "prediction": jaund_first.get("predicted_class", jaund_pred),
                "probability": jaund_first.get("confidence", jaundice_result.get("probability"))
            }

        # Quality details
        qual_dict = {}
        if quality_result:
            qual_dict = {
                "is_acceptable": quality_result.get("is_acceptable", True),
                "blur_score": quality_result.get("blur_score"),
                "brightness_score": quality_result.get("brightness_score")
            }

        # Tissue coverage
        seg_dict = {}
        if segmentation_result:
            seg_dict = {
                "tissue_coverage_pct": segmentation_result.get("tissue_coverage_pct", 0.0),
                "detected_regions": segmentation_result.get("detected_regions", [])
            }

        return {
            "demographic_reference": {
                "age_years": age_years,
                "clinical_sex": clinical_sex,
                "is_pregnant": is_pregnant,
                "pregnancy_trimester": pregnancy_trimester
            },
            "anemia_screening": anemia_dict,
            "clinical_decision": decision_dict,
            "slid_screening": slid_dict,
            "jaundice_screening": jaundice_dict,
            "image_quality": qual_dict,
            "segmentation": seg_dict
        }

    def _call_openai_vision(
        self,
        client: Any,
        sanitized_payload: Dict[str, Any],
        b64_image: Optional[str]
    ) -> Dict[str, Any]:
        """Executes structured chat completion with vision input."""

        system_prompt = (
            "You are the EyeHealthAI Specialized Research Vision Evaluator. You provide objective, "
            "rigorous, non-diagnostic personalized interpretations of ocular screening photographs.\n\n"
            "STRICT CLINICAL RULES:\n"
            "1. You are an investigational AI tool, NOT a licensed clinician or diagnostic device.\n"
            "2. Never claim a definitive diagnosis. Frame all findings as observational screening indicators.\n"
            "3. Ground your explanation strictly in the provided quantitative model outputs and WHO 2024 hemoglobin thresholds.\n"
            "4. Demographics (age, clinical sex, pregnancy) must be explained ONLY in terms of how they set the clinical reference cutoff.\n"
            "5. If the patient is <18 or >88, explicitly state that the model was trained on adults (18-88) and extrapolation requires clinical caution.\n"
            "6. Always require confirmatory clinical tests (e.g. Venous CBC, Slit-lamp exam).\n"
            "7. Return PURE JSON adhering strictly to the schema below without markdown code blocks."
        )

        user_prompt_text = (
            f"Please analyze the following sanitized screening findings and image:\n\n"
            f"Screening Data:\n{json.dumps(sanitized_payload, indent=2)}\n\n"
            "Return a JSON object with this exact structure:\n"
            "{\n"
            '  "bilingual": {\n'
            '    "en": {\n'
            '      "overall_summary": "2-3 sentence personalized screening summary for the research participant.",\n'
            '      "findings_by_target": {\n'
            '        "hemoglobin": "Detailed observation on palpebral conjunctiva pallor/erythema and continuous Hb estimate.",\n'
            '        "conjunctival_surface": "Assessment of visible conjunctival injection or redness (SLID).",\n'
            '        "scleral_icterus": "Assessment of visible scleral yellowing (categorical proxy, not serum bilirubin)."\n'
            '      },\n'
            '      "patient_profile_context": "Explanation of how patient age and clinical sex determine WHO 2024 diagnostic cutoff (e.g. 12.0 or 13.0 g/dL), noting demographics are never fed into vision neural networks.",\n'
            '      "confidence_and_uncertainty": "Empirical model uncertainty, 95% CI bounds, and MAE context.",\n'
            '      "image_quality_and_reliability": "Impact of tissue coverage, lighting, and camera sharpness on reliability.",\n'
            '      "recommended_next_steps": ["Confirmatory venous blood CBC", "Ophthalmic slit-lamp biomicroscopy if symptomatic", "Physician review"],\n'
            '      "important_limitations": "Investigational research algorithm, not a diagnostic medical device.",\n'
            '      "visual_observations": "Combined visual description for report compatibility.",\n'
            '      "reference_context": "WHO 2024 cutoff explanation.",\n'
            '      "population_status": "Statement on whether the patient is within validated population (18-88).",\n'
            '      "model_limitations": "Model uncertainty summary.",\n'
            '      "actionable_next_steps": ["Confirmatory venous blood CBC"],\n'
            '      "summary": "Overall summary."\n'
            '    },\n'
            '    "hi": {\n'
            '      "overall_summary": "मरीज/अनुसंधान प्रतिभागी के लिए 2-3 वाक्यों का संक्षिप्त सारांश।",\n'
            '      "findings_by_target": {\n'
            '        "hemoglobin": "कंजंक्टाइवा के रंग और अनुमानित हीमोग्लोबिन का विवरण।",\n'
            '        "conjunctival_surface": "कंजंक्टाइवल लालिमा व सूजन (SLID) का मूल्यांकन।",\n'
            '        "scleral_icterus": "नेत्र श्वेतपटल के पीलेपन का श्रेणीबद्ध मूल्यांकन।"\n'
            '      },\n'
            '      "patient_profile_context": "मरीज की आयु और लिंग के अनुसार WHO 2024 हीमोग्लोबिन मानक की व्याख्या।",\n'
            '      "confidence_and_uncertainty": "मॉडल की अनिश्चितता और 95% विश्वास अंतराल।",\n'
            '      "image_quality_and_reliability": "छवि स्पष्टता और कवरेज का प्रभाव।",\n'
            '      "recommended_next_steps": ["शिरापरक रक्त सीबीसी (CBC) परीक्षण कराएं", "नेत्र विशेषज्ञ से जांच कराएं"],\n'
            '      "important_limitations": "यह केवल अनुसंधान उपकरण है, कोई अंतिम निदान नहीं।",\n'
            '      "visual_observations": "नेत्र श्लेष्मलाशोथ और श्वेतपटल का दृश्य विवरण।",\n'
            '      "reference_context": "WHO 2024 मानक संदर्भ।",\n'
            '      "population_status": "मान्यता प्राप्त आयु वर्ग (18-88 वर्ष) के संबंध में स्थिति।",\n'
            '      "model_limitations": "मॉडल सीमाएं।",\n'
            '      "actionable_next_steps": ["सीबीसी (CBC) परीक्षण कराएं"],\n'
            '      "summary": "संक्षिप्त सारांश।"\n'
            '    }\n'
            '  }\n'
            "}"
        )

        content_items = [{"type": "text", "text": user_prompt_text}]
        if b64_image:
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}",
                    "detail": "low"
                }
            })

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_items}
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.2
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)

        bilingual = parsed.get("bilingual", {})
        raw_en = bilingual.get("en", {})
        raw_hi = bilingual.get("hi", {})

        def _format_section_dict(data: dict, default_summary: str, is_hindi: bool = False):
            findings = data.get("findings_by_target")
            if not isinstance(findings, dict):
                findings = {
                    "hemoglobin": data.get("visual_observations", ""),
                    "conjunctival_surface": "",
                    "scleral_icterus": ""
                }
            steps = data.get("recommended_next_steps") or data.get("actionable_next_steps") or []
            if isinstance(steps, str):
                steps = [steps]
            summary_txt = data.get("overall_summary") or data.get("summary") or default_summary

            return {
                "overall_summary": summary_txt,
                "findings_by_target": findings,
                "patient_profile_context": data.get("patient_profile_context") or data.get("reference_context", ""),
                "confidence_and_uncertainty": data.get("confidence_and_uncertainty") or data.get("model_limitations", ""),
                "image_quality_and_reliability": data.get("image_quality_and_reliability", ""),
                "recommended_next_steps": steps if steps else ([
                    "नियमित प्रयोगशाला से शिरापरक रक्त सीबीसी (CBC) परीक्षण कराएं।"
                ] if is_hindi else [
                    "Consult a physician for confirmatory venous blood CBC."
                ]),
                "important_limitations": data.get("important_limitations") or data.get("model_limitations", ""),
                # Retain backward compatible aliases
                "summary": summary_txt,
                "visual_observations": data.get("visual_observations") or findings.get("hemoglobin", ""),
                "reference_context": data.get("reference_context") or data.get("patient_profile_context", ""),
                "population_status": data.get("population_status", ""),
                "model_limitations": data.get("model_limitations") or data.get("important_limitations", ""),
                "actionable_next_steps": steps if steps else (["सीबीसी परीक्षण"] if is_hindi else ["Confirmatory CBC"])
            }

        return {
            "bilingual": {
                "en": _format_section_dict(raw_en, "Screening evaluation completed.", is_hindi=False),
                "hi": _format_section_dict(raw_hi, "स्क्रीनिंग मूल्यांकन पूर्ण हुआ।", is_hindi=True)
            }
        }

    def _generate_deterministic_explanation(
        self,
        age_years: Optional[int],
        clinical_sex: str,
        is_pregnant: bool,
        pregnancy_trimester: Optional[int],
        anemia_result: Optional[Dict[str, Any]],
        slid_result: Optional[Dict[str, Any]],
        jaundice_result: Optional[Dict[str, Any]],
        quality_result: Optional[Dict[str, Any]],
        segmentation_result: Optional[Dict[str, Any]],
        clinical_decision: Optional[Dict[str, Any]],
        reason: str
    ) -> Dict[str, Any]:
        """
        Authoritative local deterministic clinical rule generator.
        Provides zero-hallucination bilingual explanations aligned with WHO 2024.
        """
        # Demographic references
        sex_label_en = "Not Disclosed"
        sex_label_hi = "अघोषित"
        if clinical_sex == "female":
            sex_label_en = "Female"
            sex_label_hi = "महिला"
            if is_pregnant:
                trim_str = f" (Trimester {pregnancy_trimester})" if pregnancy_trimester else ""
                sex_label_en += f", Pregnant{trim_str}"
                sex_label_hi += f", गर्भवती"
        elif clinical_sex == "male":
            sex_label_en = "Male"
            sex_label_hi = "पुरुष"

        age_str_en = f"{age_years} years" if age_years is not None else "Age unrecorded"
        age_str_hi = f"{age_years} वर्ष" if age_years is not None else "आयु अनिर्धारित"

        # WHO Reference & Population
        ref_cutoff = 12.0
        ref_group_en = "General Adult Baseline"
        ref_group_hi = "सामान्य वयस्क मानक"
        pop_validated = True
        reason_code = None
        pop_warning_en = "Patient age falls within the model's clinically validated cohort (18–88 years)."
        pop_warning_hi = "रोगी की आयु मॉडल के चिकित्सकीय रूप से सत्यापित समूह (18-88 वर्ष) के अंतर्गत है।"

        if age_years is not None:
            if age_years < 18:
                pop_validated = False
                reason_code = "pediatric_under_18"
                pop_warning_en = (
                    f"CAUTION: Patient age ({age_years} yrs) is under 18 (Pediatric). "
                    "The deep learning model was trained strictly on adult cohorts (18–88 years). "
                    "Extrapolation to pediatric ocular vasculature is unvalidated."
                )
                pop_warning_hi = (
                    f"सावधानी: रोगी की आयु ({age_years} वर्ष) 18 वर्ष से कम (बाल चिकित्सा) है। "
                    "डीप लर्निंग मॉडल केवल वयस्क समूह (18-88 वर्ष) पर प्रशिक्षित है।"
                )
            elif age_years > 88:
                pop_validated = False
                reason_code = "geriatric_over_88"
                pop_warning_en = (
                    f"CAUTION: Patient age ({age_years} yrs) is above 88 (Geriatric). "
                    "The model's training distribution capped at age 88; microvascular senility may affect estimate."
                )
                pop_warning_hi = (
                    f"सावधानी: रोगी की आयु ({age_years} वर्ष) 88 वर्ष से अधिक है। "
                    "मॉडल का प्रशिक्षण 88 वर्ष तक के डेटा पर हुआ है।"
                )

        if clinical_decision:
            ref_cutoff = clinical_decision.get("reference_threshold_hb", 12.0)
            ref_group_en = clinical_decision.get("clinical_reference_group", ref_group_en)
            pop_val = clinical_decision.get("population_validation") or {}
            if isinstance(pop_val, dict):
                pop_validated = pop_val.get("is_validated_population", pop_validated)
                reason_code = pop_val.get("reason", reason_code)
            elif isinstance(pop_val, str):
                pop_validated = "outside" not in pop_val.lower()
                reason_code = pop_val
            if not pop_validated:
                if reason_code == "pediatric_under_18":
                    pop_warning_en = (
                        f"CAUTION: Patient age ({age_years} yrs) is under 18 (Pediatric). "
                        "The deep learning model was trained strictly on adult cohorts (18–88 years). "
                        "Extrapolation to pediatric ocular vasculature is unvalidated."
                    )
                    pop_warning_hi = (
                        f"सावधानी: रोगी की आयु ({age_years} वर्ष) 18 वर्ष से कम (बाल चिकित्सा) है। "
                        "डीप लर्निंग मॉडल केवल वयस्क समूह (18-88 वर्ष) पर प्रशिक्षित है।"
                    )
                elif reason_code == "geriatric_over_88":
                    pop_warning_en = (
                        f"CAUTION: Patient age ({age_years} yrs) is above 88 (Geriatric). "
                        "The model's training distribution capped at age 88; microvascular senility may affect estimate."
                    )
                    pop_warning_hi = (
                        f"सावधानी: रोगी की आयु ({age_years} वर्ष) 88 वर्ष से अधिक है। "
                        "मॉडल का प्रशिक्षण 88 वर्ष तक के डेटा पर हुआ है।"
                    )

        # Hb & Anemia observations
        hb_val = None
        hb_str = "Pending / Unavailable"
        anemia_cat_en = "Indeterminate"
        anemia_cat_hi = "अनिर्धारित"
        ci_str = "±2.05 g/dL"

        if anemia_result:
            hb_val = anemia_result.get("predicted_hb_g_dl") or anemia_result.get("view_mean")
            if hb_val is None and anemia_result.get("prediction"):
                preds = anemia_result.get("prediction")
                if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
                    hb_val = preds[0].get("hb_pred_g_dl") or preds[0].get("predicted_hb_g_dl")
            if hb_val is not None:
                hb_val = round(float(hb_val), 1)
                hb_str = f"{hb_val} g/dL"
            ci_95 = anemia_result.get("confidence_interval_95")
            if not ci_95 and clinical_decision:
                ci_95 = clinical_decision.get("confidence_interval_95")
            if isinstance(ci_95, dict) and "display" in ci_95:
                ci_str = ci_95["display"]
            elif isinstance(ci_95, (list, tuple)) and len(ci_95) == 2:
                ci_str = f"[{ci_95[0]:.1f} – {ci_95[1]:.1f}] g/dL"

        if clinical_decision:
            anemia_cat_en = clinical_decision.get("stage_label_en") or clinical_decision.get("clinical_category", "Indeterminate")
            anemia_cat_hi = clinical_decision.get("stage_label_hi") or "अनिर्धारित"
            if not anemia_cat_hi or anemia_cat_hi == "अनिर्धारित":
                if "Severe" in anemia_cat_en:
                    anemia_cat_hi = "गंभीर एनीमिया संकेतक"
                elif "Moderate" in anemia_cat_en:
                    anemia_cat_hi = "मध्यम एनीमिया संकेतक"
                elif "Mild" in anemia_cat_en:
                    anemia_cat_hi = "हल्का एनीमिया संकेतक"
                elif "Borderline" in anemia_cat_en:
                    anemia_cat_hi = "सीमांत / संदेहास्पद संकेतक"
                elif "Normal" in anemia_cat_en or "Non-Anemic" in anemia_cat_en:
                    anemia_cat_hi = "सामान्य सीमा के भीतर"

        # Conjunctivitis (SLID)
        slid_status = "Negative / Normal"
        slid_status_hi = "सामान्य (कोई सूजन या लालिमा नहीं)"
        if slid_result and slid_result.get("status") in ("success", "available_research_only"):
            slid_pred = slid_result.get("prediction")
            slid_first = (slid_pred[0] if isinstance(slid_pred, list) and len(slid_pred) > 0 and isinstance(slid_pred[0], dict)
                          else (slid_pred if isinstance(slid_pred, dict) else {}))
            pred_class = str(slid_first.get("predicted_class", slid_pred or "")).lower()
            conf = slid_first.get("confidence", slid_result.get("probability", 0.0))
            conf_pct = float(conf) * 100 if conf is not None else 0.0
            if "injection" in pred_class or "conjunctivitis" in pred_class:
                slid_status = f"Elevated Conjunctival Injection Risk (Probability: {conf_pct:.1f}%)"
                slid_status_hi = f"कंजंक्टाइवल लालिमा का उच्च जोखिम (संभावना: {conf_pct:.1f}%)"
            elif "normal" in pred_class or "control" in pred_class:
                slid_status = f"Normal / Non-Injected (Probability: {conf_pct:.1f}%)"
                slid_status_hi = f"सामान्य / कोई लालिमा नहीं (संभावना: {conf_pct:.1f}%)"

        # Jaundice / Icterus
        jaundice_status = "Normal / Scleral icterus not detected"
        jaundice_status_hi = "सामान्य / पीलिया के लक्षण नहीं दिखे"
        if jaundice_result and jaundice_result.get("status") in ("success", "available_research_only"):
            jaund_pred = jaundice_result.get("prediction")
            jaund_first = (jaund_pred[0] if isinstance(jaund_pred, list) and len(jaund_pred) > 0 and isinstance(jaund_pred[0], dict)
                           else (jaund_pred if isinstance(jaund_pred, dict) else {}))
            pred_class = str(jaund_first.get("predicted_class", jaund_pred or "")).lower()
            conf = jaund_first.get("confidence", jaundice_result.get("probability", 0.0))
            conf_pct = float(conf) * 100 if conf is not None else 0.0
            if "jaundic" in pred_class or "icterus" in pred_class:
                jaundice_status = f"Scleral Icterus Observed (Probability: {conf_pct:.1f}%)"
                jaundice_status_hi = f"श्वेतपटल पीलिया संकेतक (संभावना: {conf_pct:.1f}%)"
            elif "normal" in pred_class:
                jaundice_status = f"Normal / Non-Icteric Sclera (Probability: {conf_pct:.1f}%)"
                jaundice_status_hi = f"सामान्य / पीलिया नहीं दिखा (संभावना: {conf_pct:.1f}%)"

        # Quality & Tissue
        cov_pct = 0.0
        if segmentation_result:
            cov_pct = segmentation_result.get("tissue_coverage_pct", 0.0)

        # Assemble English Explanation
        en_obs = (
            f"Ocular computer vision analysis scanned the palpebral conjunctiva and bulbar sclera "
            f"(tissue coverage: {cov_pct:.1f}%). Estimated hemoglobin is {hb_str} (Empirical 95% CI: {ci_str}). "
            f"Conjunctival redness/SLID assessment: {slid_status}. Scleral icterus screening: {jaundice_status}."
        )
        en_ref = (
            f"Under WHO 2024 guidelines, for profile [{sex_label_en}, {age_str_en}], "
            f"the diagnostic threshold for anemia is {ref_cutoff:.1f} g/dL ({ref_group_en}). "
            f"The current screening category is evaluated as: {anemia_cat_en}."
        )
        en_lim = (
            f"Investigational research algorithm. Optical estimates can be influenced by ambient lighting, "
            f"device sensor calibration, and conjunctival micro-vascular variations. "
            f"Mean Absolute Error (MAE): 1.48 g/dL."
        )
        en_actions = [
            "Obtain a confirmatory laboratory Venous Blood Complete Blood Count (CBC) with automated hematology analyzer.",
            "If conjunctival injection or ocular discomfort is present, schedule an in-person slit-lamp biomicroscopy.",
            "Share this research report with your attending physician or ophthalmologist."
        ]
        if not pop_validated:
            en_actions.insert(0, "Note: Patient is outside the model's validated age range (18–88 yrs); do not rely on non-invasive Hb estimate.")

        en_summary = (
            f"Patient ({age_str_en}, {sex_label_en}) screened with estimated Hb of {hb_str} vs WHO cutoff {ref_cutoff:.1f} g/dL. "
            f"Status: {anemia_cat_en}. Confirmatory venous laboratory testing is required."
        )

        # Assemble Hindi Explanation
        hi_obs = (
            f"नेत्र कंप्यूटर विजन ने पुतली की निचली झिल्ली (conjunctiva) और सफेद भाग (sclera) का विश्लेषण किया "
            f"(कवरेज: {cov_pct:.1f}%)। अनुमानित हीमोग्लोबिन {hb_str} (95% विश्वास अंतराल: {ci_str}) है। "
            f"कंजक्टिवाइटिस स्थिति: {slid_status_hi}। पीलिया स्थिति: {jaundice_status_hi}।"
        )
        hi_ref = (
            f"WHO 2024 दिशानिर्देशों के तहत, [{sex_label_hi}, {age_str_hi}] के लिए "
            f"एनीमिया की नैदानिक सीमा {ref_cutoff:.1f} g/dL निर्धारित है। "
            f"वर्तमान स्क्रीनिंग मूल्यांकन: {anemia_cat_hi}।"
        )
        hi_lim = (
            "यह एक प्रायोगिक शोध उपकरण है, कोई अंतिम निदान नहीं। परिवेशीय रोशनी और कैमरे की गुणवत्ता से अनुमान प्रभावित हो सकता है। "
            "औसत निरपेक्ष त्रुटि (MAE): 1.48 g/dL।"
        )
        hi_actions = [
            "नियमित प्रयोगशाला से शिरापरक रक्त सीबीसी (CBC) परीक्षण कराकर हीमोग्लोबिन की पुष्टि करें।",
            "आंख में लालिमा, जलन या खुजली होने पर नेत्र विशेषज्ञ (Ophthalmologist) से जांच कराएं।",
            "यह रिपोर्ट केवल प्रारंभिक स्क्रीनिंग है, इसे अपने चिकित्सक को दिखाएं।"
        ]
        if not pop_validated:
            hi_actions.insert(0, "ध्यान दें: रोगी की आयु मॉडल के मान्य वर्ग (18-88 वर्ष) से बाहर है; इस अनुमान पर पूर्ण निर्भरता न रखें।")

        hi_summary = (
            f"रोगी ({age_str_hi}, {sex_label_hi}) का अनुमानित हीमोग्लोबिन {hb_str} है (WHO मानक: {ref_cutoff:.1f} g/dL)। "
            f"स्थिति: {anemia_cat_hi}। पुष्टि के लिए चिकित्सकीय रक्त परीक्षण आवश्यक है।"
        )
        # Structured Findings & Quality for English
        en_target_findings = {
            "hemoglobin": f"Palpebral conjunctiva optical inspection indicates estimated Hb of {hb_str} (95% CI: {ci_str}), classified as {anemia_cat_en}.",
            "conjunctival_surface": f"Conjunctival redness/SLID assessment: {slid_status}.",
            "scleral_icterus": f"Scleral icterus assessment: {jaundice_status} (categorical proxy, not serum bilirubin)."
        }
        en_ci_unc = f"Empirical model 95% Confidence Interval: {ci_str}. Model Mean Absolute Error (MAE): 1.48 g/dL."
        en_img_qual = f"Evaluated with tissue coverage of {cov_pct:.1f}%. Optimal illumination and proper ocular eversion are essential for optical reliability."

        # Structured Findings & Quality for Hindi
        hi_target_findings = {
            "hemoglobin": f"कंजंक्टाइवा के ऑप्टिकल विश्लेषण से अनुमानित हीमोग्लोबिन {hb_str} (95% विश्वास अंतराल: {ci_str}) है, स्थिति: {anemia_cat_hi}।",
            "conjunctival_surface": f"कंजंक्टाइवल लालिमा (SLID) मूल्यांकन: {slid_status_hi}।",
            "scleral_icterus": f"श्वेतपटल पीलिया मूल्यांकन: {jaundice_status_hi} (यह केवल श्रेणीबद्ध स्क्रीनिंग है, रक्त बिलीरुबिन नहीं)।"
        }
        hi_ci_unc = f"मॉडल 95% विश्वास अंतराल: {ci_str}। औसत निरपेक्ष त्रुटि (MAE): 1.48 g/dL।"
        hi_img_qual = f"नेत्र संरचना कवरेज: {cov_pct:.1f}%। सटीक ऑप्टिकल अनुमान के लिए उचित प्रकाश व्यवस्था और फोकस अनिवार्य है।"

        return {
            "status": "completed",
            "provider": "local_deterministic",
            "model_used": "WHO_2024_Deterministic_Rules",
            "fallback_reason": reason,
            "external_ai_consent": False,
            "bilingual": {
                "en": {
                    "overall_summary": en_summary,
                    "findings_by_target": en_target_findings,
                    "patient_profile_context": f"{en_ref} {pop_warning_en}",
                    "confidence_and_uncertainty": en_ci_unc,
                    "image_quality_and_reliability": en_img_qual,
                    "recommended_next_steps": en_actions,
                    "important_limitations": en_lim,
                    # Backward-compatible fields
                    "visual_observations": en_obs,
                    "reference_context": en_ref,
                    "population_status": pop_warning_en,
                    "model_limitations": en_lim,
                    "actionable_next_steps": en_actions,
                    "summary": en_summary
                },
                "hi": {
                    "overall_summary": hi_summary,
                    "findings_by_target": hi_target_findings,
                    "patient_profile_context": f"{hi_ref} {pop_warning_hi}",
                    "confidence_and_uncertainty": hi_ci_unc,
                    "image_quality_and_reliability": hi_img_qual,
                    "recommended_next_steps": hi_actions,
                    "important_limitations": hi_lim,
                    # Backward-compatible fields
                    "visual_observations": hi_obs,
                    "reference_context": hi_ref,
                    "population_status": pop_warning_hi,
                    "model_limitations": hi_lim,
                    "actionable_next_steps": hi_actions,
                    "summary": hi_summary
                }
            }
        }
