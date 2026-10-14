"""
src/inference/doctor_buddy.py
-----------------------------
Context-Aware AI Doctor Mascot Service for EyeHealthAI.

Provides deterministic, clinical-grade bilingual guidance (English/Hindi)
paired with Gemini LLM enrichment when available.

MANDATORY PRIVACY & CLINICAL SAFETY GUARDRAILS:
  1. NEVER transmit raw images, camera frames, or base64 pixels to the LLM.
  2. NEVER forward individual patient PII (names, phone numbers, emails, DOB) to the LLM.
  3. NEVER forward numerical lab predictions or invent medical diagnoses.
  4. Only sanitized state events, contextual tags, and user clarifications are processed.
  5. Deterministic clinical rules ALWAYS override LLM outputs for capture validation,
     quality gates, and model blocking reasons.
"""

import os
import re
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic Bilingual Guidance Knowledge Base
# ---------------------------------------------------------------------------

DETERMINISTIC_EVENTS = {
    "welcome": {
        "en": {
            "message": "Hello! I'm AI Dr. Astra, your AI Eye Health Guide. Let's begin by selecting your preferred language and reviewing the screening steps.",
            "speak_text": "Hello! I am Doctor Astra, your AI Eye Health Guide. Let's begin by selecting your language.",
            "expression": "happy",
            "severity": "info",
            "suggested_next_step": "Select English or Hindi to continue."
        },
        "hi": {
            "message": "नमस्ते! मैं एआई डॉ. एस्ट्रा हूँ, आपका एआई नेत्र स्वास्थ्य गाइड। अपनी पसंदीदा भाषा चुनकर आगे बढ़ें।",
            "speak_text": "नमस्ते! मैं डॉक्टर एस्ट्रा हूँ, आपका एआई नेत्र स्वास्थ्य गाइड। अपनी पसंदीदा भाषा चुनकर आगे बढ़ें।",
            "expression": "happy",
            "severity": "info",
            "suggested_next_step": "जारी रखने के लिए हिंदी या अंग्रेज़ी चुनें।"
        }
    },
    "registration": {
        "en": {
            "message": "Please review the participant consent terms. EyeHealthAI is an investigational research prototype — your images are processed strictly in RAM and never retained permanently.",
            "speak_text": "Please review the consent terms. EyeHealthAI is an investigational prototype and processes images strictly in memory.",
            "expression": "default",
            "severity": "info",
            "suggested_next_step": "Fill details and accept research consent to continue."
        },
        "hi": {
            "message": "कृपया सहमति शर्तों की समीक्षा करें। यह एक अनुसंधान प्रोटोटाइप है — आपकी तस्वीरें केवल रैम में प्रोसेस होती हैं और डिस्क पर कभी सुरक्षित नहीं रखी जातीं।",
            "speak_text": "कृपया सहमति शर्तों की समीक्षा करें। आपकी तस्वीरें केवल रैम में प्रोसेस होती हैं और कभी स्थायी रूप से सुरक्षित नहीं रखी जातीं।",
            "expression": "default",
            "severity": "info",
            "suggested_next_step": "विवरण भरें और अनुसंधान सहमति स्वीकार करें।"
        }
    },
    "preparation": {
        "en": {
            "message": "Ensure you are in soft, diffuse lighting without direct glare or heavy shadows. Position yourself comfortably about 10 to 15 centimeters from the lens.",
            "speak_text": "Ensure you are in soft, diffuse lighting. Position yourself about 10 to 15 centimeters from the camera.",
            "expression": "default",
            "severity": "info",
            "suggested_next_step": "Review the 6 preparation steps and click Ready to Capture."
        },
        "hi": {
            "message": "कमरे में सामान्य, अप्रत्यक्ष रोशनी रखें। तेज धूप या फ्लैश से बचें और कैमरे से 10 से 15 सेंटीमीटर की आरामदायक दूरी बनाएं।",
            "speak_text": "कमरे में सामान्य रोशनी रखें और कैमरे से 10 से 15 सेंटीमीटर की दूरी बनाएं।",
            "expression": "default",
            "severity": "info",
            "suggested_next_step": "तैयारी के 6 चरणों की समीक्षा करें और कैप्चर बटन दबाएं।"
        }
    },
    "camera_permission_prompt": {
        "en": {
            "message": "Your browser will ask for camera permission. Please click 'Allow' so we can guide your eye alignment in real time.",
            "speak_text": "Your browser will ask for camera permission. Please click Allow.",
            "expression": "alert",
            "severity": "info",
            "suggested_next_step": "Click 'Allow' in your browser's camera prompt."
        },
        "hi": {
            "message": "आपका ब्राउज़र कैमरे की अनुमति मांगेगा। आंख के सही संरेखण के लिए कृपया 'Allow' या अनुमति दें पर क्लिक करें।",
            "speak_text": "ब्राउज़र में कैमरे की अनुमति दें ताकि हम आंख की स्थिति जांच सकें।",
            "expression": "alert",
            "severity": "info",
            "suggested_next_step": "ब्राउज़र में 'Allow' पर क्लिक करें।"
        }
    },
    "camera_permission_denied": {
        "en": {
            "message": "Camera access was blocked. To fix this: click the lock/camera icon in your browser URL bar, set Camera to 'Allow', and refresh this page. Alternatively, use File Upload Mode.",
            "speak_text": "Camera access was blocked. Please enable camera permissions in your browser address bar or use file upload mode.",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "Enable camera permission in browser settings or switch to File Upload."
        },
        "hi": {
            "message": "कैमरा अनुमति अस्वीकृत हो गई है। इसे ठीक करने के लिए: ब्राउज़र एड्रेस बार में लॉक या कैमरा आइकन पर क्लिक करें, अनुमति दें, और पेज रिफ्रेश करें। या फोटो अपलोड मोड का उपयोग करें।",
            "speak_text": "कैमरा अनुमति अस्वीकृत है। ब्राउज़र एड्रेस बार में अनुमति दें या फोटो अपलोड मोड का उपयोग करें।",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "ब्राउज़र सेटिंग्स में कैमरा चालू करें या फोटो अपलोड मोड चुनें।"
        }
    },
    "camera_not_found": {
        "en": {
            "message": "No webcam device was found on this system. Please connect an external webcam or switch to File Upload Mode to analyze existing clinical images.",
            "speak_text": "No webcam was detected. Please connect a camera or switch to File Upload mode.",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "Switch to File Upload Mode."
        },
        "hi": {
            "message": "सिस्टम पर कोई वेबकैम नहीं मिला। कृपया कैमरा कनेक्ट करें या फोटो अपलोड मोड का चयन करें।",
            "speak_text": "कोई कैमरा नहीं मिला। कृपया फोटो अपलोड मोड का उपयोग करें।",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "फोटो अपलोड मोड में स्विच करें।"
        }
    },
    "positioning_too_far": {
        "en": {
            "message": "You are too far from the camera. Move closer until one eye fills the green tracking box (macro close-up at 10–15 cm). Avoid full-face selfies.",
            "speak_text": "Please move closer to the camera. We need a close-up of one eye, not a full face.",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "Move closer so one eye fills the reticle."
        },
        "hi": {
            "message": "आप कैमरे से बहुत दूर हैं। कैमरे के करीब आएं ताकि एक आंख हरे बॉक्स में स्पष्ट दिखे (10-15 सेमी की दूरी)। पूरे चेहरे की सेल्फी न लें।",
            "speak_text": "कृपया कैमरे के करीब आएं। हमें केवल एक आंख की नजदीकी तस्वीर चाहिए।",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "कैमरे के करीब आएं ताकि आंख फ्रेम में पूरी आ सके।"
        }
    },
    "positioning_aligned": {
        "en": {
            "message": "Perfect alignment! Keep one eye steady inside the reticle and ensure the palpebral conjunctiva or sclera is visible. Hold steady and capture.",
            "speak_text": "Perfect alignment. Hold steady and click Capture Still.",
            "expression": "happy",
            "severity": "success",
            "suggested_next_step": "Click Capture Still."
        },
        "hi": {
            "message": "उत्कृष्ट स्थिति! आंख को फ्रेम में स्थिर रखें और निचली पलक या स्क्लेरा को स्पष्ट रखें। स्थिर रहकर फोटो खींचें।",
            "speak_text": "बिल्कुल सही स्थिति। स्थिर रहें और फोटो खींचें।",
            "expression": "happy",
            "severity": "success",
            "suggested_next_step": "'कैप्चर स्टिल' पर क्लिक करें।"
        }
    },
    "quality_blurry": {
        "en": {
            "message": "The captured frame is blurry (sharpness score too low). Please hold the device steady, allow the lens to focus, and recapture.",
            "speak_text": "The image is blurry. Please hold steady and recapture.",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "Click Recapture and hold still."
        },
        "hi": {
            "message": "तस्वीर धुंधली है। कृपया हाथ स्थिर रखें, कैमरे को फोकस करने दें, और दोबारा फोटो खींचें।",
            "speak_text": "तस्वीर धुंधली है। कृपया हाथ स्थिर रखकर दोबारा फोटो लें।",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "दोबारा फोटो लें और स्थिर रहें।"
        }
    },
    "quality_dark": {
        "en": {
            "message": "The image is too dark. Increase room illumination or face an indirect soft light source to illuminate the conjunctival microvasculature.",
            "speak_text": "The image is too dark. Please face a light source and recapture.",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "Improve lighting and recapture."
        },
        "hi": {
            "message": "तस्वीर बहुत अंधेरी है। कमरे की रोशनी बढ़ाएं या रोशनी की ओर मुंह करके दोबारा फोटो खींचें।",
            "speak_text": "तस्वीर में रोशनी कम है। कृपया रोशनी की ओर मुंह करके दोबारा फोटो लें।",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "रोशनी बढ़ाएं और दोबारा फोटो खींचें।"
        }
    },
    "quality_bright": {
        "en": {
            "message": "Glare or over-exposure detected. Turn off direct camera flash and step back from harsh direct reflections.",
            "speak_text": "Too much glare detected. Avoid direct flash or sunlight reflection.",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "Reduce direct glare and recapture."
        },
        "hi": {
            "message": "अत्यधिक चमक या फ्लैश का परावर्तन मिला है। तेज फ्लैश बंद करें और सीधी धूप से बचें।",
            "speak_text": "अत्यधिक चमक मिली है। फ्लैश बंद करें और दोबारा फोटो लें।",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "चमक कम करें और दोबारा फोटो खींचें।"
        }
    },
    "quality_full_face_rejected": {
        "en": {
            "message": "Full-face selfie detected. The model requires an isolated macro eye view. Please move closer (10–15 cm) to frame only one eye.",
            "speak_text": "A full face was detected. Please move closer for a close-up macro eye view.",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "Recapture with a close-up of only one eye."
        },
        "hi": {
            "message": "पूरे चेहरे की तस्वीर मिली है। मॉडल को केवल आंख की नजदीकी तस्वीर चाहिए। कृपया 10-15 सेमी करीब आकर केवल एक आंख फ्रेम करें।",
            "speak_text": "पूरे चेहरे की तस्वीर मिली है। कृपया करीब आकर केवल एक आंख की फोटो लें।",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "करीब आएं और केवल एक आंख की फोटो लें।"
        }
    },
    "quality_passed": {
        "en": {
            "message": "Optical quality verified! Sharpness and luminance pass clinical criteria. You can now run the research inference pipeline.",
            "speak_text": "Image quality verified. You can now run the screening analysis.",
            "expression": "happy",
            "severity": "success",
            "suggested_next_step": "Click 'Run Research Inference'."
        },
        "hi": {
            "message": "तस्वीर की गुणवत्ता प्रमाणित! तीखापन और रोशनी सही है। अब आप विश्लेषण शुरू कर सकते हैं।",
            "speak_text": "गुणवत्ता की पुष्टि हो गई है। अब आप विश्लेषण शुरू कर सकते हैं।",
            "expression": "happy",
            "severity": "success",
            "suggested_next_step": "'विश्लेषण चलाएं' पर क्लिक करें।"
        }
    },
    "model_mendeley_blocked": {
        "en": {
            "message": "The legacy Mendeley conjunctivitis model is safely blocked due to a resolution confound in its original dataset. However, the SLID slit-lamp hyperemia model is available for valid research screening.",
            "speak_text": "The legacy Mendeley model is blocked due to dataset shortcuts. Please use the SLID model for research screening.",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "Select the SLID Conjunctival Injection module."
        },
        "hi": {
            "message": "पुराना मेंडेली कंजंक्टिवाइटिस मॉडल डेटासेट शॉर्टकट के कारण ब्लॉक है। वैध अनुसंधान के लिए एसएलआईडी (SLID) मॉडल का उपयोग करें।",
            "speak_text": "पुराना मॉडल डेटासेट त्रुटि के कारण ब्लॉक है। कृपया एसएलआईडी मॉडल चुनें।",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "एसएलआईडी कंजंक्टाइवल इंजेक्शन मॉड्यूल चुनें।"
        }
    },
    "model_webcam_blocked": {
        "en": {
            "message": "Camera domain validation gate active. Models trained on clinical photography are not validated on consumer laptop webcams by default. Researchers may enable the experimental override if needed.",
            "speak_text": "Webcam inference is restricted by default because the model was trained on clinical photography.",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "Use uploaded clinical photos or enable the researcher experimental override."
        },
        "hi": {
            "message": "कैमरा डोमेन सुरक्षा सक्रिय है। क्लिनिकल फोटोग्राफी पर प्रशिक्षित मॉडल लैपटॉप वेबकैम पर डिफ़ॉल्ट रूप से प्रतिबंधित हैं। शोधकर्ता प्रायोगिक ओवरराइड सक्षम कर सकते हैं।",
            "speak_text": "वेबकैम जांच प्रतिबंधित है क्योंकि मॉडल क्लिनिकल तस्वीरों पर प्रशिक्षित था।",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "अपलोड मोड का उपयोग करें या प्रायोगिक ओवरराइड चेकबॉक्स चुनें।"
        }
    },
    "results_anemia_borderline": {
        "en": {
            "message": "Your hemoglobin estimate is near the diagnostic boundary. In this equivocal zone, optical estimation has higher uncertainty. A laboratory CBC venous blood test is strongly recommended.",
            "speak_text": "The estimate is near the diagnostic boundary. A confirmatory laboratory CBC blood test is recommended.",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "Consult a healthcare professional for a confirmatory CBC test."
        },
        "hi": {
            "message": "अनुमानित हीमोग्लोबिन नैदानिक सीमा के करीब है। इस क्षेत्र में अनिश्चितता अधिक होती है। प्रयोगशाला सीबीसी (CBC) रक्त परीक्षण की दृढ़ता से अनुशंसा की जाती है।",
            "speak_text": "अनुमान सीमा के करीब है। प्रयोगशाला सीबीसी रक्त परीक्षण कराने की सलाह दी जाती है।",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "पुष्टि के लिए योग्य चिकित्सक और सीबीसी परीक्षण की सलाह लें।"
        }
    },
    "results_ready": {
        "en": {
            "message": "Screening analysis complete! Review your clinical score, confidence intervals, and chromatic biomarkers. Download the official 2-page report below.",
            "speak_text": "Analysis complete. You can now review your scorecard and download the report.",
            "expression": "happy",
            "severity": "success",
            "suggested_next_step": "Download the 2-page PDF report or review findings."
        },
        "hi": {
            "message": "जांच पूरी हो गई है! अपना स्कोरकार्ड, विश्वास अंतराल, और क्रोमैटिक बायोमार्कर देखें। आप 2-पेज की विस्तृत रिपोर्ट डाउनलोड कर सकते हैं।",
            "speak_text": "जांच पूरी हुई। अपना परिणाम देखें और 2-पेज की रिपोर्ट डाउनलोड करें।",
            "expression": "happy",
            "severity": "success",
            "suggested_next_step": "2-पृष्ठों की पीडीएफ रिपोर्ट डाउनलोड करें।"
        }
    }
}


class DoctorBuddyEventRequest(BaseModel):
    """Sanitized event request sent from frontend to Doctor Buddy."""
    event_code: str = Field(..., max_length=64, description="Deterministic event code e.g. 'positioning_too_far'")
    language: str = Field("en", max_length=5, description="'en' or 'hi'")
    context_note: Optional[str] = Field(None, max_length=256, description="Optional non-PII contextual tag")


class DoctorBuddyResponse(BaseModel):
    """Validated Doctor Buddy response schema."""
    message: str
    speak_text: str
    expression: str  # 'default', 'speaking', 'alert', 'happy'
    severity: str    # 'info', 'warning', 'success', 'danger'
    suggested_next_step: str
    mode: str        # 'deterministic' or 'gemini_enriched'


def get_doctor_buddy_guidance(
    event_code: str,
    language: str = "en",
    context_note: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieves deterministic guidance for an application event.
    Guarantees strict fallback and zero hallucinations.
    """
    clean_lang = "hi" if language.lower().startswith("hi") else "en"
    clean_event = event_code.strip().lower()

    event_data = DETERMINISTIC_EVENTS.get(clean_event)
    if not event_data:
        # Generic fallback
        default_en = {
            "message": "I'm Dr. Netra. Let's proceed carefully with the next step of your ocular screening.",
            "speak_text": "I am Doctor Netra. Let's proceed with the next step.",
            "expression": "default",
            "severity": "info",
            "suggested_next_step": "Follow the on-screen instructions."
        }
        default_hi = {
            "message": "मैं डॉ. नेत्रा हूँ। आइए अपनी नेत्र जांच के अगले चरण पर ध्यान दें।",
            "speak_text": "मैं डॉक्टर नेत्रा हूँ। आइए अगले चरण पर ध्यान दें।",
            "expression": "default",
            "severity": "info",
            "suggested_next_step": "स्क्रीन पर दिए गए निर्देशों का पालन करें।"
        }
        chosen = default_hi if clean_lang == "hi" else default_en
    else:
        chosen = event_data.get(clean_lang, event_data.get("en"))

    return {
        "message": chosen["message"],
        "speak_text": chosen["speak_text"],
        "expression": chosen["expression"],
        "severity": chosen["severity"],
        "suggested_next_step": chosen["suggested_next_step"],
        "mode": "deterministic"
    }


async def ask_doctor_buddy_question(
    question: str,
    language: str = "en",
    current_step: int = 1,
    session_id: Optional[str] = None,
    history: Optional[List[dict]] = None
) -> Dict[str, Any]:
    """
    Processes a user-asked question through OpenAI / Gemini LLM if available,
    with complete PII/medical-advice safety filtering and deterministic session-grounded fallback.
    """
    from src.inference.copilot import sanitize_input, is_blocked_topic, _get_gemini_client, _SYSTEM_INSTRUCTION
    from src.config import settings

    clean_lang = "hi" if language.lower().startswith("hi") else "en"

    # Step 1: Check blocked clinical advice
    if is_blocked_topic(question):
        blocked_msg_en = (
            "I cannot provide a personal medical diagnosis or prescribe treatment. "
            "I can only explain how this research screening prototype works and guide your camera capture. "
            "Please consult a certified physician for medical advice."
        )
        blocked_msg_hi = (
            "मैं व्यक्तिगत चिकित्सीय निदान या दवा का सुझाव नहीं दे सकता। "
            "मैं केवल इस अनुसंधान प्रोटोटाइप की प्रक्रिया और कैमरा मार्गदर्शन में आपकी मदद कर सकता हूँ। "
            "कृपया चिकित्सीय सलाह हेतु योग्य डॉक्टर से संपर्क करें।"
        )
        return {
            "message": blocked_msg_hi if clean_lang == "hi" else blocked_msg_en,
            "speak_text": "I cannot provide personal medical diagnoses. Please consult a doctor." if clean_lang == "en" else "मैं व्यक्तिगत चिकित्सीय निदान नहीं दे सकता। कृपया डॉक्टर से संपर्क करें।",
            "expression": "alert",
            "severity": "warning",
            "suggested_next_step": "Consult a healthcare professional for clinical concerns.",
            "mode": "blocked"
        }

    # Step 2: Sanitize input
    sanitized_q = sanitize_input(question)
    q_low = sanitized_q.lower()

    # Step 3: Fetch active session context if session_id is provided
    session_context_str = ""
    session_info = {}
    if session_id:
        try:
            from src.database.database import SessionLocal
            from src.database.crud import get_session_by_id
            with SessionLocal() as db:
                db_s = get_session_by_id(db, session_id)
                if db_s:
                    q_sum = db_s.quality_summary or {}
                    session_info["session_id"] = db_s.id
                    session_info["quality_summary"] = q_sum
                    session_info["results"] = {}
                    for r in db_s.model_results:
                        session_info["results"][r.module] = {
                            "status": r.operational_status,
                            "prediction_value": r.prediction_value,
                            "unit": r.unit
                        }
                    # Build contextual string for LLM or rule engine
                    res_parts = []
                    for m, d in session_info["results"].items():
                        res_parts.append(f"{m.upper()}: status={d['status']}, value={d['prediction_value']} {d.get('unit','')}")
                    seg_cov = q_sum.get("segmentation", {}).get("coverage_pct", {})
                    sec_vis = q_sum.get("secondary_vision", {}).get("summary", "")
                    cal_ens = q_sum.get("calibrated_ensemble", {}).get("tier_label", "")
                    session_context_str = (
                        f"ACTIVE SCREENING SESSION CONTEXT (Session ID: {db_s.id}):\n"
                        f"- Module Outputs: {'; '.join(res_parts)}\n"
                        f"- Segmentation Coverage: Conjunctiva={seg_cov.get('conjunctiva', 'N/A')}%, Sclera={seg_cov.get('sclera', 'N/A')}%\n"
                        f"- Calibrated Screening Tier: {cal_ens}\n"
                        f"- Secondary AI Visual Findings: {sec_vis}\n"
                    )
        except Exception as e:
            logger.debug(f"Failed to fetch session context for Doctor Buddy: {e}")

    # Step 4: Try OpenAI Chat API (Official SDK, configurable model, grounded session context)
    openai_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", None)
    if openai_key and len(openai_key.strip()) > 5:
        try:
            import openai
            o_client = openai.OpenAI(api_key=openai_key.strip(), timeout=15.0)
            o_model = os.getenv("OPENAI_MODEL_NAME") or getattr(settings, "OPENAI_MODEL_NAME", "gpt-4o-mini")

            system_instruction = (
                f"{_SYSTEM_INSTRUCTION}\n\n"
                f"You are AI Dr. Astra, the expert ocular health research guide and companion for the EyeHealthAI (OcuScan) clinical screening framework. "
                f"When asked about this project or how it works, always refer to EyeHealthAI and explain its three independent clinical models: Model A (ResNet-50 Anemia), Model B (MobileNetV3 SLID hyperemia), and Model C (MobileNetV3 Scleral Icterus). "
                f"Respond in {'Hindi (Devanagari script)' if clean_lang == 'hi' else 'English'}. "
                f"Ground your answer strictly in the patient's verified screening session findings below. "
                f"Explain medical concepts in compassionate, simple language. "
                f"Never fabricate diseases, laboratory hemoglobin or bilirubin numbers, confidence percentages, or diagnostic claims. "
                f"Always emphasize that this is investigational screening and advise consulting a physician with a confirmatory CBC blood test.\n\n"
                f"{session_context_str}"
            )

            messages = [{"role": "system", "content": system_instruction}]

            # Contextual multi-turn follow-up history (last 4 turns)
            if history and isinstance(history, list):
                for h in history[-4:]:
                    r = "user" if h.get("role") == "user" else "assistant"
                    t = sanitize_input(str(h.get("content", "")))
                    if t:
                        messages.append({"role": r, "content": t})

            messages.append({"role": "user", "content": sanitized_q})

            chat_resp = o_client.chat.completions.create(
                model=o_model,
                messages=messages,
                max_tokens=600,
                temperature=0.3
            )

            resp_text = chat_resp.choices[0].message.content or ""
            if resp_text.strip():
                # Provide contextual follow-up chips
                follow_ups = [
                    "What should I discuss with my doctor?",
                    "Why is a laboratory CBC blood test needed?"
                ] if clean_lang != "hi" else [
                    "मुझे अपने डॉक्टर से क्या पूछना चाहिए?",
                    "सीबीसी रक्त परीक्षण क्यों आवश्यक है?"
                ]

                return {
                    "message": resp_text.strip(),
                    "speak_text": re.sub(r'[*#_`]', '', resp_text)[:180],
                    "expression": "speaking",
                    "severity": "info",
                    "suggested_next_step": "Share this screening report with your doctor.",
                    "mode": "openai_chat",
                    "provider": "openai",
                    "model": o_model,
                    "follow_up_suggestions": follow_ups
                }
        except Exception as e:
            logger.warning(f"OpenAI Doctor Buddy chat candidate encountered an error ({e}). Attempting fallback...")

    # Step 5: Try Gemini API if OpenAI is unavailable
    client = _get_gemini_client()
    candidate_models = [
        getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"),
        "gemini-flash-latest",
        "gemini-2.5-flash-lite",
        "gemini-1.5-flash",
    ]

    if client:
        system_prompt = (
            f"{_SYSTEM_INSTRUCTION}\n\n"
            f"You are AI Dr. Astra, an expert ocular health guide and research companion. "
            f"The user is at Step {current_step} of the screening workflow. "
            f"Respond in {'Hindi (Devanagari script)' if clean_lang == 'hi' else 'English'}. "
            f"Keep answers clear, highly informative, structured with markdown, and strictly within safety boundaries.\n"
            f"{session_context_str}"
        )
        contents = []
        if history and isinstance(history, list):
            for h in history[-4:]:
                role = "user" if h.get("role") == "user" else "model"
                h_text = sanitize_input(str(h.get("content", "")))
                if h_text:
                    contents.append({"role": role, "parts": [{"text": h_text}]})
        contents.append({"role": "user", "parts": [{"text": sanitized_q}]})

        for target_model in candidate_models:
            try:
                response = client.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config={
                        "system_instruction": system_prompt,
                        "temperature": 0.3,
                        "max_output_tokens": 512,
                    }
                )
                resp_text = response.text.strip() if response.text else ""
                if resp_text:
                    return {
                        "message": resp_text,
                        "speak_text": re.sub(r'[*#_`]', '', resp_text)[:180],
                        "expression": "speaking",
                        "severity": "info",
                        "suggested_next_step": "Continue with your screening step.",
                        "mode": "gemini_enriched"
                    }
            except Exception as e:
                logger.debug(f"Gemini candidate {target_model} failed: {e}")

    # -------------------------------------------------------------------------
    # Patient-Centric Question Handlers (Session-Grounded & Plain Language)
    # -------------------------------------------------------------------------

    # 1. "What does my result mean?"
    if any(k in q_low for k in ["what does my result mean", "result mean", "meaning of my result", "what do these results mean",
                                "explain my result", "what does this mean", "मेरे परिणाम का क्या मतलब", "परिणाम का अर्थ", "रिजल्ट का मतलब"]):
        hb_val_str = "N/A"
        hb_num = None
        tier_str = "Screening In Progress"
        slid_status = "Normal"
        jaund_status = "Normal"
        if session_info and session_info.get("results"):
            res_dict = session_info["results"]
            anemia_res = res_dict.get("anemia", {})
            hb_val = anemia_res.get("prediction_value")
            try:
                hb_num = float(hb_val) if hb_val is not None else None
                hb_val_str = f"{hb_num:.2f} g/dL"
            except (ValueError, TypeError):
                pass
            slid_status = res_dict.get("slid", {}).get("status", "Normal")
            jaund_status = res_dict.get("jaundice", {}).get("status", "Normal")
            cal = session_info.get("quality_summary", {}).get("calibrated_ensemble", {})
            tier_str = cal.get("tier_label", "BORDERLINE / CLINICAL REVIEW")

        if clean_lang == "hi":
            if hb_num is not None:
                if hb_num < 11.0:
                    hb_meaning = f"आपके हीमोग्लोबिन का अनुमान **{hb_val_str}** है। यह निचली पलक की रक्त वाहिकाओं में हल्का या मध्यम पीलापन (pallor) दर्शाता है, जो संभावित एनीमिया का संकेत हो सकता है।"
                else:
                    hb_meaning = f"आपके हीमोग्लोबिन का अनुमान **{hb_val_str}** है, जो सामान्य स्वस्थ संदर्भ सीमा के भीतर है। पलक की रक्त वाहिकाओं में पर्याप्त लालिमा देखी गई है।"
            else:
                hb_meaning = "आपके हीमोग्लोबिन की जांच का मूल्यांकन चल रहा है या परिणाम उपलब्ध नहीं है।"

            return {
                "message": (
                    f"### आपके परिणामों का सरल अर्थ (Screening Summary)\n\n"
                    f"1. 🩸 **रक्त हीमोग्लोबिन (Hb)**: {hb_meaning}\n"
                    f"2. 👁️ **आंख की सतह (कंजंक्टिवा)**: स्थिति **{slid_status}**। यह देखा गया कि क्या आंख की सतह पर कोई असामान्य लालिमा या सूजन है।\n"
                    f"3. ☀️ **आंख का सफेद भाग (स्क्लेरा)**: स्थिति **{jaund_status}**। आंख के सफेद भाग में किसी पीलेपन (पीलिया/इक्टेरस) की जांच की गई।\n"
                    f"4. 📊 **समग्र स्क्रीनिंग श्रेणी**: **{tier_str}**।\n\n"
                    f"💡 **यह क्या साबित नहीं कर सकता**: यह जांच केवल फोटो के रंगों का गैर-आक्रामक अनुमान है। यह कोई अंतिम डॉक्टरी निदान नहीं है। कमरे की रोशनी और कैमरे के कारण रंगों में थोड़ा अंतर आ सकता है।"
                ),
                "speak_text": f"आपके परिणामों का अर्थ: हीमोग्लोबिन अनुमान {hb_val_str} है। यह केवल एक गैर-आक्रामक जांच है, अंतिम निदान हेतु डॉक्टर से परामर्श करें।",
                "expression": "speaking",
                "severity": "info",
                "suggested_next_step": "डॉक्टर से सीबीसी रक्त परीक्षण पर चर्चा करें।",
                "mode": "patient_guidance_meaning"
            }
        else:
            if hb_num is not None:
                if hb_num < 8.0:
                    hb_meaning = f"Estimated at **{hb_val_str}** (noticeable pallor in the microvascular vessels of your inner eyelid, suggesting possible lower red blood cell levels)."
                elif hb_num < 11.0:
                    hb_meaning = f"Estimated at **{hb_val_str}** (mild-to-moderate pallor observed in the inner eyelid lining, which can be associated with mild anemia)."
                elif hb_num < 12.0:
                    hb_meaning = f"Estimated at **{hb_val_str}** (borderline normal range; healthy pinkish coloration with slight variation)."
                else:
                    hb_meaning = f"Estimated at **{hb_val_str}** (healthy baseline microvascular coloration with good vascular perfusion)."
            else:
                hb_meaning = "Hemoglobin estimation awaiting completed scan or not recorded."

            return {
                "message": (
                    f"### What Your Screening Results Mean (In Simple Terms)\n\n"
                    f"1. 🩸 **Blood Hemoglobin**: {hb_meaning}\n"
                    f"2. 👁️ **Eye Surface Health**: Status is **{slid_status}**. We screened the surface of your eye for signs of visible redness or irritation.\n"
                    f"3. ☀️ **Eye Whites (Sclera)**: Status is **{jaund_status}**. We screened for yellowish coloration that could be linked to bile pigments.\n"
                    f"4. 📊 **Overall Screening Category**: **{tier_str}**.\n\n"
                    f"💡 **Important Context & Diagnostic Boundaries**:\n"
                    f"- This screening estimates **surface optical coloration**, not internal blood cells directly.\n"
                    f"- It is an early research check, but **cannot definitively diagnose or rule out any condition**.\n"
                    f"- Ambient lighting, camera sensors, and individual eyelid characteristics influence readings, which is why clinical laboratory confirmation is always recommended."
                ),
                "speak_text": f"Your estimated hemoglobin is {hb_val_str}. This non-invasive screening reflects surface eyelid coloration and should be confirmed with a standard blood test.",
                "expression": "speaking",
                "severity": "info",
                "suggested_next_step": "Schedule a routine visit with your physician to discuss these findings.",
                "mode": "patient_guidance_meaning"
            }

    # 2. "Why do I need another test?"
    if any(k in q_low for k in ["why do i need another test", "why another test", "why need another test", "why do i need a cbc",
                                "why cbc test", "why blood test", "why confirmatory", "दूसरा टेस्ट", "दूसरे टेस्ट", "रक्त परीक्षण क्यों", "सीबीसी क्यों", "अन्य टेस्ट", "पुष्टि की जरूरत", "परीक्षण की आवश्यकता"]):
        if clean_lang == "hi":
            return {
                "message": (
                    "### आपको दूसरा परीक्षण (जैसे सीबीसी रक्त जांच) क्यों करवाना चाहिए?\n\n"
                    "1. **सतह बनाम आंतरिक रक्त**: EyeHealthAI केवल आपकी पलक और आंख की सतह पर रोशनी के परावर्तन और रंग का विश्लेषण करता है। यह रक्त वाहिकाओं के अंदर जाकर कोशिकाओं को सीधे नहीं गिन सकता।\n"
                    "2. **नैदानिक सटीकता (Gold Standard)**: एक प्रयोगशाला सीबीसी (Complete Blood Count) नस से रक्त लेकर माइक्रोस्कोप और हेमोकाउंटर से हीमोग्लोबिन की ±0.1 g/dL तक सटीक माप करता है, जबकि कैमरा अनुमान ±1.5 g/dL तक भिन्न हो सकता है।\n"
                    "3. **रोशनी और कैमरे का प्रभाव**: कमरे की रोशनी, स्क्रीन ब्राइटनेस और फोन/वेबकैम के रंग सेंसर फोटो के रंग को थोड़ा बदल सकते हैं।\n"
                    "4. **आपकी सुरक्षा सर्वोपरि**: कोई भी डॉक्टर केवल फोटो देखकर दवा या आयरन की खुराक शुरू नहीं करता; प्रयोगशाला रिपोर्ट से सही और सुरक्षित इलाज सुनिश्चित होता है।"
                ),
                "speak_text": "कंप्यूटर विज़न केवल आंख की सतह का रंग देखता है। सटीक मात्रा और सुरक्षित उपचार के लिए लैब सीबीसी रक्त परीक्षण आवश्यक होता है।",
                "expression": "thoughtful",
                "severity": "info",
                "suggested_next_step": "नियमित स्वास्थ्य जांच के दौरान डॉक्टर से सीबीसी टेस्ट का अनुरोध करें।",
                "mode": "patient_guidance_why_test"
            }
        else:
            return {
                "message": (
                    "### Why a Follow-up Laboratory Test is Recommended\n\n"
                    "1. 🔬 **Surface Optical Patterns vs. Internal Blood Cells**:\n"
                    "   EyeHealthAI analyzes light reflecting off the mucosal surface of your eyelid and eye whites. While tissue redness strongly correlates with hemoglobin, camera sensors cannot directly count blood cells inside your bloodstream.\n\n"
                    "2. ⚖️ **Certified Laboratory Precision (Gold Standard)**:\n"
                    "   A standard Complete Blood Count (CBC) uses a small venous blood sample analyzed by clinical laboratory hemocounters with high precision (accurate to ±0.1 g/dL, whereas non-invasive optical screening has an expected uncertainty of ±1.5 g/dL).\n\n"
                    "3. 💡 **Lighting and Device Variations**:\n"
                    "   Ambient room lighting, camera white balance, and skin undertones can subtly alter photograph colors. A laboratory blood panel is completely immune to external optical conditions.\n\n"
                    "4. 🛡️ **Patient Safety and Actionable Care**:\n"
                    "   Medical guidelines require verified laboratory results before a physician recommends treatments, dietary changes, or iron supplementation."
                ),
                "speak_text": "A standard laboratory blood test measures red blood cells directly inside your blood, ensuring precise and safe clinical care.",
                "expression": "thoughtful",
                "severity": "info",
                "suggested_next_step": "Bring your screening report to your next medical appointment.",
                "mode": "patient_guidance_why_test"
            }

    # 3. "Which results still need confirmation?"
    if any(k in q_low for k in ["which results still need confirmation", "which result needs confirmation", "what needs confirmation",
                                "unconfirmed", "need confirmation", "which needs checking", "किन परिणामों की पुष्टि", "पुष्टि आवश्यक", "पुष्टि की जानी बाकी", "पुष्टि चाहिए", "जांच की पुष्टि", "पुष्टि"]):
        if clean_lang == "hi":
            return {
                "message": (
                    "### किन परिणामों की डॉक्टरी पुष्टि आवश्यक है?\n\n"
                    "सभी एआई-आधारित गैर-आक्रामक परिणाम अनुसंधान अनुमान हैं और निम्नलिखित पुष्टियों की आवश्यकता होती है:\n\n"
                    "- 🩸 **हीमोग्लोबिन / एनीमिया**: इसे एक प्रमाणित नस के रक्त परीक्षण (**Complete Blood Count - CBC**) द्वारा पुष्टि की आवश्यकता है।\n"
                    "- 👁️ **आंख की लालिमा (कंजंक्टिवाइटिस)**: यदि आंख में खुजली, जलन, दर्द या पानी आ रहा हो, तो एक नेत्र विशेषज्ञ द्वारा **स्लिट-लैंप बायोमाइक्रोस्कोपी** परीक्षण आवश्यक है।\n"
                    "- ☀️ **पीलिया / स्क्लेरा का पीलापन**: यदि आंख के सफेद हिस्से में पीलापन दिखे, तो डॉक्टर से **लिवर फंक्शन टेस्ट (LFT / Serum Bilirubin)** की सलाह लें।\n"
                    "- 📷 **कैमरा गुणवत्ता सीमाएं**: सामान्य वेबकैम तस्वीरों की तुलना में अस्पताल के क्लिनिकल कैमरे अधिक संवेदनशील होते हैं।"
                ),
                "speak_text": "हीमोग्लोबिन के लिए सीबीसी रक्त परीक्षण, और आंख की लालिमा के लिए नेत्र विशेषज्ञ की जांच आवश्यक पुष्टि कदम हैं।",
                "expression": "speaking",
                "severity": "info",
                "suggested_next_step": "अपनी 2-पेज पीडीएफ रिपोर्ट डाउनलोड कर डॉक्टर को दिखाएं।",
                "mode": "patient_guidance_confirmation"
            }
        else:
            return {
                "message": (
                    "### Which Results Need Medical Confirmation?\n\n"
                    "Because this is an experimental non-invasive screening tool, **all three screening targets require standard medical verification** before making any health decisions:\n\n"
                    "1. 🩸 **Hemoglobin & Anemia Estimation**:\n"
                    "   - *Needs*: A standard venous **Complete Blood Count (CBC)** blood test.\n"
                    "   - *Why*: Direct measurement of red blood cell count, hematocrit, and hemoglobin concentration.\n\n"
                    "2. 👁️ **Eye Surface Redness & Irritation (Conjunctivitis)**:\n"
                    "   - *Needs*: An in-person **slit-lamp examination** by an optometrist or ophthalmologist if redness, itching, discharge, or discomfort is present.\n"
                    "   - *Why*: Differentiates dry eyes, allergies, bacterial, or viral conjunctivitis.\n\n"
                    "3. ☀️ **Scleral Yellowness (Jaundice)**:\n"
                    "   - *Needs*: A routine **Serum Bilirubin (Total & Direct)** blood panel.\n"
                    "   - *Why*: Our model screens visual yellowness categorically; only a laboratory blood test provides true clinical bilirubin concentrations."
                ),
                "speak_text": "All three targets benefit from confirmation: a CBC blood test for hemoglobin, an eye exam for redness, and a bilirubin panel for yellowing.",
                "expression": "speaking",
                "severity": "info",
                "suggested_next_step": "Consult your primary care physician or eye specialist.",
                "mode": "patient_guidance_confirmation"
            }

    # 4. "What should I discuss with my doctor?"
    if any(k in q_low for k in ["what should i discuss with my doctor", "discuss with my doctor", "talk to my doctor",
                                "what to ask doctor", "what should i ask my doctor", "questions for doctor", "ask doctor",
                                "डॉक्टर से क्या पूछूं", "डॉक्टर से क्या बात करें", "डॉक्टर को क्या बताएं", "डॉक्टर से क्या चर्चा", "डॉक्टर से चर्चा"]):
        hb_mention = ""
        if session_info and session_info.get("results"):
            hb_val = session_info["results"].get("anemia", {}).get("prediction_value")
            if hb_val:
                hb_mention = f" (specifically mentioning the estimated {hb_val} g/dL from your screening)"

        if clean_lang == "hi":
            return {
                "message": (
                    "### डॉक्टर से परामर्श के लिए व्यावहारिक चेकलिस्ट\n\n"
                    "जब आप अपने डॉक्टर से मिलें, तो आप निम्नलिखित प्रश्न पूछ सकते हैं:\n\n"
                    "1. 📋 **स्क्रीनिंग रिपोर्ट दिखाएं**: 'डॉक्टर साहब, मैंने एक शोध-आधारित नेत्र जांच करवाई है जिसमें यह रिपोर्ट मिली है।'\n"
                    "2. 🩸 **सीबीसी टेस्ट का अनुरोध**: 'क्या आप मेरे हीमोग्लोबिन और आयरन के स्तर की पुष्टि के लिए एक साधारण सीबीसी (CBC) रक्त परीक्षण लिख सकते हैं?'\n"
                    "3. 🩺 **अपने लक्षण बताएं**: यदि आपको हाल ही में थकान, चक्कर, सिरदर्द, सांस फूलना या ठंड लगना महसूस हुआ हो, तो डॉक्टर को अवश्य बताएं।\n"
                    "4. 👁️ **आंखों के आराम की सलाह**: यदि आपकी आंखें सूखी, थकी या लाल महसूस होती हैं, तो स्क्रीन उपयोग और लूब्रिकेंट आई-ड्रॉप्स के बारे में पूछें। बिना पर्ची के कोई भी औषधीय ड्रॉप न डालें।"
                ),
                "speak_text": "डॉक्टर को अपनी रिपोर्ट दिखाएं, सीबीसी रक्त परीक्षण का अनुरोध करें, और थकान या कमजोरी जैसे किसी भी लक्षण की चर्चा करें।",
                "expression": "happy",
                "severity": "info",
                "suggested_next_step": "रिपोर्ट को प्रिंट करें या अपने फोन में सेव करके डॉक्टर को दिखाएं।",
                "mode": "patient_guidance_doctor_questions"
            }
        else:
            return {
                "message": (
                    "### What to Discuss With Your Healthcare Provider\n\n"
                    "Here is a friendly, structured checklist you can take to your appointment:\n\n"
                    f"1. 📄 **Share Your Screening Report**:\n"
                    f"   Hand your doctor the 2-Page PDF (especially **Page 2: Clinician & Research Review**, which contains the technical model details and confidence intervals).\n\n"
                    f"2. 🩸 **Request a Confirmatory Blood Test**:\n"
                    f"   *\"I recently participated in a non-invasive eye screening that estimated my hemoglobin{hb_mention}. Could we order a routine Complete Blood Count (CBC) to verify my blood levels?\"*\n\n"
                    f"3. 💬 **Mention Any Everyday Symptoms**:\n"
                    f"   Be sure to tell your doctor if you have experienced unusual tiredness, cold hands/feet, shortness of breath on stairs, or dizzy spells.\n\n"
                    f"4. 👁️ **Discuss Eye Comfort & Screen Habits**:\n"
                    f"   If your eyes frequently feel dry, gritty, or strained, ask your doctor or optometrist about appropriate eye rest habits or lubricating drops. Never use medicated steroid drops without a prescription."
                ),
                "speak_text": "Take your 2-page report to your doctor, ask about a routine CBC blood test, and discuss any tiredness or eye fatigue you may feel.",
                "expression": "happy",
                "severity": "info",
                "suggested_next_step": "Download your official 2-Page PDF report to bring to your appointment.",
                "mode": "patient_guidance_doctor_questions"
            }

    # Knowledge Base: Active Session Findings & Detailed Model Observations
    if any(k in q_low for k in ["what did each", "observe", "findings", "disagree", "confidence mean", "quality sufficient",
                                "uncertain", "what test", "what additional", "test is needed", "segmentation", "reference match",
                                "क्या देखा", "मॉडल ने क्या", "कॉन्फिडेंस", "क्वालिटी", "कौनसा टेस्ट"]):
        # If session_info is available, generate grounded explanation of the active session
        if session_info and session_info.get("results"):
            res_dict = session_info["results"]
            anemia_res = res_dict.get("anemia", {})
            slid_res = res_dict.get("slid", {})
            jaund_res = res_dict.get("jaundice", {})
            q_sum = session_info.get("quality_summary", {})
            seg = q_sum.get("segmentation", {})
            cov = seg.get("coverage_pct", {})
            cal = q_sum.get("calibrated_ensemble", {})

            hb_str = f"{anemia_res.get('prediction_value', 'N/A')} g/dL" if anemia_res.get("prediction_value") else "Unavailable"
            slid_str = f"Status: {slid_res.get('status', 'N/A')}"
            jaund_str = f"Status: {jaund_res.get('status', 'N/A')}"

            if clean_lang == "hi":
                return {
                    "message": (
                        f"### आपके सक्रिय स्क्रीनिंग सत्र के परिणाम (Session: {session_info.get('session_id', 'Active')[:12]})\n\n"
                        f"1. **मॉडल ए (एनीमिया)**: अनुमानित हीमोग्लोबिन **{hb_str}** (निचली पलक के रंग और एरिथेमा इंडेक्स से)।\n"
                        f"2. **मॉडल बी (कंजंक्टिवाइटिस)**: स्लिट-लैंप मोबाइलनेट **{slid_str}**।\n"
                        f"3. **मॉडल सी (पीलिया / स्क्लेरा)**: स्क्लेरल इक्टेरस **{jaund_str}**।\n"
                        f"4. **एआई सेगमेंटेशन कवरेज**: कंजेक्टिवा **{cov.get('conjunctiva', 'N/A')}%**, स्क्लेरा **{cov.get('sclera', 'N/A')}%**।\n"
                        f"5. **कैलिब्रेटेड एन्सेम्बल टीयर**: **{cal.get('tier_label', 'Borderline Review')}**।\n\n"
                        f"⚠️ **पुष्टि हेतु आवश्यक परीक्षण**: गैर-आक्रामक विज़ुअल अनुमान की पुष्टि के लिए मानक सीबीसी (CBC) रक्त परीक्षण और कुल सीरम बिलीरुबिन लैब टेस्ट आवश्यक है।"
                    ),
                    "speak_text": f"आपके सत्र में हीमोग्लोबिन {hb_str} अनुमानित हुआ है। चिकित्सीय पुष्टि हेतु सीबीसी रक्त परीक्षण आवश्यक है।",
                    "expression": "speaking",
                    "severity": "info",
                    "suggested_next_step": "विस्तृत 2-पृष्ठों की पीडीएफ रिपोर्ट डाउनलोड करें।",
                    "mode": "deterministic_session_grounded"
                }
            else:
                return {
                    "message": (
                        f"### Active Session Clinical Screening Findings (Session: {session_info.get('session_id', 'Active')[:12]})\n\n"
                        f"- **Model A (Anemia Regressor)**: Predicted Hemoglobin **{hb_str}** derived from palpebral conjunctiva pallor and Erythema Index.\n"
                        f"- **Model B (SLID Conjunctivitis)**: {slid_str} (evaluated for anterior ocular surface vascular injection).\n"
                        f"- **Model C (Scleral Icterus)**: {jaund_str} (evaluated for scleral yellowness index).\n"
                        f"- **Ocular Segmentation**: Successfully localized **{cov.get('conjunctiva', 'N/A')}%** palpebral conjunctiva and **{cov.get('sclera', 'N/A')}%** exposed sclera.\n"
                        f"- **Calibrated Ensemble Tier**: **{cal.get('tier_label', 'BORDERLINE / CLINICAL REVIEW')}**.\n\n"
                        f"🔍 **Why Models May Report Different Findings**: Each model inspects an entirely independent anatomical region with specialized chromatic loss functions. For example, severe anemia affects mucosal microvasculature without altering scleral yellowness.\n\n"
                        f"🔬 **Recommended Confirmatory Tests**: Standard venous CBC blood count with automated differential and serum total/direct bilirubin."
                    ),
                    "speak_text": f"In your active session, estimated hemoglobin is {hb_str}. Clinical confirmation requires standard venous blood testing.",
                    "expression": "speaking",
                    "severity": "info",
                    "suggested_next_step": "Review the full findings and download the clinical PDF report.",
                    "mode": "deterministic_session_grounded"
                }

    # Knowledge Base: Project Overview & How it works
    if any(k in q_low for k in ["what is this project", "project overview", "how does it work", "about this project",
                                "tell me about this project", "explain this project", "what can you do", "who are you",
                                "what is eyehealthai", "overview", "kaise kaam karta", "kya hai", "batao", "system overview",
                                "project details", "working", "how it works"]):
        if clean_lang == "hi":
            return {
                "message": (
                    "### EyeHealthAI — नैदानिक वास्तुकला अवलोकन\n\n"
                    "EyeHealthAI एक बहु-मॉडल कंप्यूटर विज़न अनुसंधान प्रणाली है जो तीन प्रमुख नेत्र स्वास्थ्य लक्ष्यों की गैर-आक्रामक जांच करती है:\n\n"
                    "1. **मॉडल ए (एनीमिया / हीमोग्लोबिन प्रतिगमन)**: निचली पलक (palpebral conjunctiva) के रंग से हीमोग्लोबिन (Hb in g/dL) का अनुमान लगाता है (ResNet-50, MAE ~1.50 g/dL)।\n"
                    "2. **मॉडल बी और बी2 (कंजंक्टिवाइटिस और हाइपरमिया)**: क्लिनिकल स्लिट-लैंप इमेज डेटासेट (SLID) पर आधारित 98.78% सटीकता वाला क्लासिफायर। पुराना मेंडेली मॉडल डेटासेट शॉर्टकट के कारण ब्लॉक है।\n"
                    "3. **मॉडल सी (पीलिया / स्क्लेरल इक्टेरस)**: आंख के सफेद हिस्से में पीलेपन (icterus) की पहचान करता है। संख्यात्मक बिलीरुबिन मान सुरक्षित रूप से रोका गया है।\n\n"
                    "**मुख्य सुरक्षा मानक**:\n"
                    "- **शून्य डेटा लीकेज**: रोगी-स्तरीय विभाजन (`GroupKFold`) से डेटा लीकेज की रोकथाम।\n"
                    "- **गोपनीयता**: तस्वीरें केवल रैम (RAM) में प्रोसेस होती हैं और डिस्क पर कभी सुरक्षित नहीं रखी जातीं।"
                ),
                "speak_text": "EyeHealthAI पलक के रंग से हीमोग्लोबिन, स्लिट-लैंप से कंजंक्टिवाइटिस, और स्क्लेरा से पीलिया की जांच करता है। आप किसी भी मॉडल या चरण के बारे में पूछ सकते हैं।",
                "expression": "happy",
                "severity": "info",
                "suggested_next_step": "अपनी जांच शुरू करने के लिए अगला चरण चुनें या किसी मॉडल के बारे में पूछें।",
                "mode": "deterministic"
            }
        else:
            return {
                "message": (
                    "### EyeHealthAI — Clinical Architecture Overview\n\n"
                    "EyeHealthAI is an investigational computer vision framework designed for non-invasive ocular health screening across three distinct targets:\n\n"
                    "1. **Model A (Anemia / Hemoglobin Regression)**: Estimates laboratory hemoglobin (Hb in g/dL) from lower palpebral conjunctiva pallor using a pretrained ResNet-50 backbone (Test MAE ~1.50 g/dL).\n"
                    "2. **Model B & B2 (Conjunctivitis & Hyperemia)**: Identifies conjunctival and ciliary injection using the clinical SLID slit-lamp dataset (MobileNetV3, 98.78% accuracy). Note: The legacy Mendeley dataset is safely blocked due to a resolution confound.\n"
                    "3. **Model C (Jaundice / Scleral Icterus)**: Screens for scleral yellowing via chromatic color-space shifts. Numerical bilirubin (mg/dL) is strictly withheld to preserve ground-truth integrity.\n\n"
                    "**Key Guardrails**:\n"
                    "- **Zero Data Leakage**: Group-aware patient splitting ensures no subject's images span both train and test partitions.\n"
                    "- **Privacy & Ethics**: In-memory RAM processing only; no images stored permanently on disk."
                ),
                "speak_text": "EyeHealthAI screens for anemia from eyelid pallor, conjunctivitis from slit-lamp injection, and jaundice from scleral yellowing. Feel free to ask about any specific model or capture step.",
                "expression": "happy",
                "severity": "info",
                "suggested_next_step": "Proceed to the next screening step or ask about any specific model.",
                "mode": "deterministic"
            }

    # Knowledge Base: Model A (Anemia & Hemoglobin Regression)
    elif any(k in q_low for k in ["anemia", "hemoglobin", "hb", "pallor", "erythema", "resnet", "regression", "mae",
                                  "who staging", "borderline", "g/dl", "blood", "एनीमिया", "हीमोग्लोबिन", "रक्त"]):
        if clean_lang == "hi":
            return {
                "message": (
                    "### मॉडल ए: एनीमिया और हीमोग्लोबिन प्रतिगमन\n\n"
                    "- **बायोमार्कर**: निचली आंतरिक पलक (palpebral conjunctiva) का पीलापन (pallor) और एरिथेमा इंडेक्स।\n"
                    "- **आर्किटेक्चर**: प्रीट्रेन्ड ResNet-50 बैकबोन जिसमें कस्टम लीनियर रिग्रेशन हेड जोड़ा गया है।\n"
                    "- **डेटासेट**: प्रामाणिक क्लिनिकल लेबोरेटरी संदर्भ स्प्रेडशीट (`India.xlsx`, `Italy.xlsx`) से जुड़े प्रयोगशाला हीमोग्लोबिन मान।\n"
                    "- **सटीकता**: टेस्ट मीन एब्सोल्यूट एरर (MAE) ~1.50 g/dL (पेशेंट-लेवल ग्रुपिंग द्वारा परीक्षित)।\n"
                    "- **WHO गंभीरता श्रेणियां**:\n"
                    "  - सामान्य (Normal): ≥ 12.0 g/dL (महिला) / ≥ 13.0 g/dL (पुरुष)\n"
                    "  - हल्का (Mild): 11.0 – 11.9 g/dL\n"
                    "  - मध्यम (Moderate): 8.0 – 10.9 g/dL\n"
                    "  - गंभीर (Severe): < 8.0 g/dL\n\n"
                    "⚠️ *क्लिनिकल चेतावनी*: यदि अनुमान नैदानिक सीमा के करीब हो (±0.85 g/dL), तो सीबीसी रक्त परीक्षण अनिवार्य है।"
                ),
                "speak_text": "एनीमिया मॉडल पलक के रंग से हीमोग्लोबिन का अनुमान लगाता है जिसका टेस्ट एरर लगभग 1.5 g/dL है। चिकित्सीय पुष्टि हेतु सीबीसी रक्त परीक्षण अनिवार्य है।",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "निचली पलक की स्पष्ट तस्वीर लेने के लिए कैप्चर गाइड का पालन करें।",
                "mode": "deterministic"
            }
        else:
            return {
                "message": (
                    "### Model A: Anemia & Hemoglobin Estimation\n\n"
                    "- **Biomarker**: Pallor and erythema index of the lower inner eyelid (palpebral conjunctiva).\n"
                    "- **Architecture**: Pretrained ResNet-50 deep CNN with custom continuous regression head.\n"
                    "- **Dataset & Ground Truth**: Paired with authentic laboratory hemoglobin measurements from clinical trial spreadsheets (`India.xlsx`, `Italy.xlsx`).\n"
                    "- **Performance**: Test Mean Absolute Error (MAE) of ~1.50 g/dL with patient-level cross-validation to prevent leakage.\n"
                    "- **WHO Severity Staging**:\n"
                    "  - Normal: ≥ 12.0 g/dL (Females) / ≥ 13.0 g/dL (Males)\n"
                    "  - Mild: 11.0 – 11.9 g/dL\n"
                    "  - Moderate: 8.0 – 10.9 g/dL\n"
                    "  - Severe: < 8.0 g/dL\n\n"
                    "⚠️ *Clinical Rule*: Borderline estimates (within ±0.85 g/dL of boundary) mandate a confirmatory laboratory CBC venous blood test."
                ),
                "speak_text": "The anemia model uses a ResNet-50 regressor on eyelid pallor with a test error of about 1.5 g/dL. Clinical confirmation requires a standard CBC blood test.",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "Follow the capture guide to expose the inner palpebral conjunctiva.",
                "mode": "deterministic"
            }

    # Knowledge Base: Model B & B2 (Conjunctivitis & Hyperemia / SLID)
    elif any(k in q_low for k in ["conjunctivitis", "pink eye", "hyperemia", "mendeley", "slid", "injection",
                                  "red eye", "redness", "कंजंक्टिवाइटिस", "आंख लाल", "गुलाबी"]):
        if clean_lang == "hi":
            return {
                "message": (
                    "### मॉडल बी और बी2: कंजंक्टिवाइटिस और हाइपरमिया जांच\n\n"
                    "- **मेंडेली शॉर्टकट ब्लॉकर**: हमारे डेटासेट ऑडिट में पाया गया कि मेंडेली डेटासेट में रिज़ॉल्यूशन का एक गंभीर भ्रम था (सामान्य छवियां: 56×56 px; कंजंक्टिवाइटिस: 352×252 px)। मॉडल बीमारी के बजाय छवि का आकार सीख रहा था, इसलिए इस पुराने मॉडल को सुरक्षित रूप से ब्लॉक कर दिया गया है।\n"
                    "- **SLID मॉडल समाधान**: हमने स्लिट-लैंप इमेज डेटासेट (SLID) के 1,480 नैदानिक नमूनों को एकीकृत किया है।\n"
                    "- **आर्किटेक्चर और सटीकता**: MobileNetV3-Small क्लासिफायर जो बिना किसी कृत्रिम विसंगति के 98.78% सटीकता से कंजंक्टाइवल इंजेक्शन की पहचान करता है।"
                ),
                "speak_text": "पुराना मेंडेली मॉडल इमेज रिज़ॉल्यूशन शॉर्टकट के कारण ब्लॉक है। हमारा सक्रिय एसएलआईडी स्लिट-लैंप मॉडल 98.8% सटीकता से कंजंक्टिवाइटिस की जांच करता है।",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "कंजंक्टिवाइटिस जांच के लिए एसएलआईडी मॉड्यूल का चयन करें।",
                "mode": "deterministic"
            }
        else:
            return {
                "message": (
                    "### Model B & B2: Conjunctivitis & Ocular Hyperemia\n\n"
                    "- **The Mendeley Shortcut Blocker**: In our dataset audit, we discovered that the Mendeley conjunctivitis dataset had a severe resolution confound (Normal images: 56×56 px; Conjunctivitis images: 352×252 px). Standard CNNs learned image dimensions rather than medical pathology. EyeHealthAI blocks this legacy model by default.\n"
                    "- **SLID Clinical Classifier (Active)**: We integrated the Slit-Lamp Image Dataset (SLID) featuring 1,480 clinically validated images.\n"
                    "- **Architecture & Accuracy**: MobileNetV3-Small classifier delivering 98.78% accuracy for bulbar and palpebral conjunctival injection without artifact bias."
                ),
                "speak_text": "The legacy Mendeley model is blocked because of an image resolution shortcut. Our SLID slit-lamp model delivers 98.8% accuracy for conjunctival injection.",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "Select the SLID Conjunctival Injection module for screening.",
                "mode": "deterministic"
            }

    # Knowledge Base: Model C (Jaundice & Scleral Icterus)
    elif any(k in q_low for k in ["jaundice", "sclera", "icterus", "yellow", "bilirubin", "liver",
                                  "पीलिया", "बिलीरुबिन", "पीला", "स्क्लेरा"]):
        if clean_lang == "hi":
            return {
                "message": (
                    "### मॉडल सी: पीलिया और स्क्लेरल इक्टेरस जांच\n\n"
                    "- **बायोमार्कर**: स्क्लेरल इक्टेरस (रक्त में बिलीरुबिन बढ़ने पर आंख के सफेद भाग में आने वाला पीलापन)।\n"
                    "- **सक्रिय जांच**: CIE-Lab कलर स्पेस और क्रोमैटिक दूरी विश्लेषण के माध्यम से श्रेणीबद्ध द्विआधारी वर्गीकरण (Normal vs. Visibly Jaundiced)।\n"
                    "- **संख्यात्मक बिलीरुबिन क्यों नहीं?**: हीमोग्लोबिन के विपरीत, खुले साहित्य में नेत्र तस्वीरों को सीरम बिलीरुबिन के प्रयोगशाला मानों से जोड़ने वाला कोई सत्यापित डेटासेट उपलब्ध नहीं है। हमारे सिद्धांत (*Never fabricate ground truth*) के अनुसार, हम बिना वास्तविक डेटासेट के मनगढ़ंत संख्याएं नहीं दिखाते।"
                ),
                "speak_text": "पीलिया मॉड्यूल आंख के सफेद हिस्से में पीलेपन की जांच करता है। प्रयोगशाला डेटासेट के अभाव में संख्यात्मक बिलीरुबिन मान प्रदर्शित नहीं किए जाते।",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "पीलिया की जांच के लिए आंख का सफेद हिस्सा स्पष्ट रूप से दिखाएं।",
                "mode": "deterministic"
            }
        else:
            return {
                "message": (
                    "### Model C: Jaundice & Scleral Icterus\n\n"
                    "- **Biomarker**: Scleral icterus (yellow pigmentation of the eye's white sclera caused by bilirubin accumulation in elastin-rich tissues).\n"
                    "- **Active Screening**: Categorical binary classification (Normal vs. Visibly Jaundiced) via CIE-Lab and chromatic distance analysis.\n"
                    "- **Why Numerical Bilirubin is Withheld**: Unlike hemoglobin, there are no publicly available clinical datasets linking eye photography directly to serum bilirubin laboratory numbers. Following our core rule (*Never fabricate ground truth*), EyeHealthAI strictly refuses to synthesize fake bilirubin numbers."
                ),
                "speak_text": "The jaundice module detects visible scleral yellowing. Numerical bilirubin values are withheld because paired lab datasets do not exist in open literature.",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "Expose the white sclera clearly for jaundice screening.",
                "mode": "deterministic"
            }

    # Knowledge Base: Data Leakage Prevention & Splitting
    elif any(k in q_low for k in ["leakage", "data leakage", "split", "patient split", "groupkfold",
                                  "session", "cross-validation", "डेटा लीकेज", "विभाजन"]):
        if clean_lang == "hi":
            return {
                "message": (
                    "### डेटा लीकेज रोकथाम और रोगी-स्तरीय विभाजन\n\n"
                    "- **समस्या**: नेत्र डेटासेट में एक ही रोगी की कई तस्वीरें होती हैं (बाएं/दाएं आंख, विभिन्न कोण)। यदि छवियों को बेतरतीब ढंग से बांटा जाए, तो एक ही रोगी की तस्वीरें ट्रेन और टेस्ट दोनों में चली जाती हैं, जिससे मॉडल की सटीकता झूठी दिखती है।\n"
                    "- **हमारा समाधान**: EyeHealthAI रोगी-स्तरीय और सत्र-स्तरीय समूहीकरण (`GroupKFold`) लागू करता है। किसी भी रोगी की 100% छवियां केवल एक ही विभाजन (ट्रेन, वेलिडेशन, या टेस्ट) में रहती हैं।"
                ),
                "speak_text": "हम रोगी-स्तरीय विभाजन लागू करते हैं जिससे एक रोगी की सभी तस्वीरें केवल एक विभाजन में रहें और डेटा लीकेज न हो।",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "डैशबोर्ड पर सत्यापन रिपोर्ट की समीक्षा करें।",
                "mode": "deterministic"
            }
        else:
            return {
                "message": (
                    "### Data Leakage Prevention & Patient Splitting\n\n"
                    "- **The Challenge**: In ocular datasets, each subject contributes multiple images (left eye, right eye, different sessions). Random shuffling causes images of the same patient to appear in both training and test sets, artificially inflating accuracy.\n"
                    "- **Our Solution**: EyeHealthAI enforces strict patient-level and session-level grouping (`GroupKFold`). 100% of a patient's images reside exclusively in either train, validation, or test partitions, ensuring true out-of-sample generalization."
                ),
                "speak_text": "We enforce strict patient-level splitting so all images from any subject stay in only one split, preventing data leakage and inflated accuracy.",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "Review the validation reports on the dashboard.",
                "mode": "deterministic"
            }

    # Knowledge Base: Preprocessing & Quality Verification Gates
    elif any(k in q_low for k in ["preprocess", "quality", "blur", "sharp", "laplacian", "dark", "bright",
                                  "light", "exposure", "illumination", "धुंधला", "रोशनी", "चमक", "क्वालिटी"]):
        if clean_lang == "hi":
            return {
                "message": (
                    "### लेयर 2: प्रीप्रोसेसिंग और ऑप्टिकल गुणवत्ता जांच\n\n"
                    "किसी भी छवि को मॉडल में भेजने से पहले 4 स्वचालित सुरक्षा परीक्षण किए जाते हैं:\n"
                    "1. **लाप्लासियन ब्लर टेस्ट**: तीखापन स्कोर > 100 होना चाहिए। धुंधली तस्वीरें अस्वीकार कर दी जाती हैं।\n"
                    "2. **रोशनी परीक्षण**: औसत ल्यूमिनेंस 40 (अत्यधिक अंधेरा) और 220 (अत्यधिक चमक/फ्लैश) के बीच होनी चाहिए।\n"
                    "3. **रंग-संरक्षण**: रंग में कृत्रिम बदलाव (Hue Jitter) पूरी तरह प्रतिबंधित है क्योंकि पीलापन और लालिमा सूक्ष्म क्रोमैटिक संकेत हैं।\n"
                    "4. **आंख का अलगाव**: पूरे चेहरे की सेल्फी अस्वीकार की जाती है; केवल एक आंख का क्लोज-अप स्वीकार्य है।"
                ),
                "speak_text": "प्रत्येक तस्वीर की तीखापन, रोशनी और रंग-संरक्षण की क्लिनिकल जांच की जाती है ताकि मॉडल को स्पष्ट डेटा मिले।",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "कैमरे को 10-15 सेमी दूर रखें और हाथ स्थिर रखें।",
                "mode": "deterministic"
            }
        else:
            return {
                "message": (
                    "### Layer 2: Preprocessing & Quality Verification\n\n"
                    "Before any frame reaches inference, it must pass 4 automated optical quality checks:\n"
                    "1. **Laplacian Blur Verification**: Sharpness score must exceed variance threshold (> 100). Blurry images are rejected.\n"
                    "2. **Illumination Verification**: Mean luminance must lie within 40 (underexposed) and 220 (overexposed/glare).\n"
                    "3. **Clinical Color Preservation**: Destructive hue jitter and aggressive saturation shifts are forbidden because pallor and icterus are delicate chromatic biomarkers.\n"
                    "4. **Macro Eye Isolation**: Full-face selfies are rejected; single-eye bounding boxes (10–15 cm) are extracted."
                ),
                "speak_text": "Every capture is checked for blur using Laplacian variance, tested for proper lighting, and processed with strict color space preservation.",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "Hold device steady at 10-15 cm under soft diffuse light.",
                "mode": "deterministic"
            }

    # Knowledge Base: Camera Capture Guidance
    elif any(k in q_low for k in ["camera", "permission", "allow", "webcam", "capture", "how to take",
                                  "distance", "position", "guide", "reticle", "कैमरा", "अनुमति", "फोटो"]):
        if clean_lang == "hi":
            return {
                "message": (
                    "### स्मार्ट कैमरा कैप्चर गाइड\n\n"
                    "1. **दूरी**: कैमरे को आंख से 10 से 15 सेंटीमीटर की दूरी पर रखें (मैक्रो क्लोज-अप)।\n"
                    "2. **रोशनी**: कमरे में सामान्य, अप्रत्यक्ष रोशनी रखें। सीधी धूप या फ्लैश से बचें।\n"
                    "3. **पलक की स्थिति (एनीमिया हेतु)**: अपनी निचली पलक को धीरे से नीचे खींचें ताकि अंदरूनी लाल भाग (conjunctiva) दिखे।\n"
                    "4. **आंख को स्थिर रखें**: आंख को हरे बॉक्स के बीच में रखें और फ्रेम हरा होने पर 'कैप्चर स्टिल' दबाएं।"
                ),
                "speak_text": "कैमरे को 10 से 15 सेंटीमीटर दूर रखें, अच्छी रोशनी में रहें, और आंख को हरे बॉक्स में स्थिर रखकर फोटो लें।",
                "expression": "happy",
                "severity": "info",
                "suggested_next_step": "कैमरा चालू करें और हरे बॉक्स में आंख संरेखित करें।",
                "mode": "deterministic"
            }
        else:
            return {
                "message": (
                    "### Smart Camera Capture Guide\n\n"
                    "1. **Distance**: Position camera 10 to 15 cm from your eye (macro close-up). Avoid full-face selfies.\n"
                    "2. **Lighting**: Ensure soft, diffuse front lighting without direct camera flash or glare.\n"
                    "3. **Eyelid Exposure (for Anemia)**: Gently pull down your lower eyelid to expose the inner mucosal conjunctiva.\n"
                    "4. **Steady Alignment**: Align one eye inside the green reticle box and click 'Capture Still' when steady."
                ),
                "speak_text": "Hold camera 10 to 15 centimeters away, ensure soft lighting without flash, and keep one eye steady inside the green reticle.",
                "expression": "happy",
                "severity": "info",
                "suggested_next_step": "Enable camera and align eye within the green tracking box.",
                "mode": "deterministic"
            }

    # Knowledge Base: Safety, Ethics & Privacy
    elif any(k in q_low for k in ["privacy", "security", "safe", "gdpr", "hipaa", "store", "save",
                                  "data", "pii", "ram", "सुरक्षा", "प्राइवेसी", "डेटा"]):
        if clean_lang == "hi":
            return {
                "message": (
                    "### क्लिनिकल सुरक्षा और डेटा गोपनीयता\n\n"
                    "- **रैम-ओनली प्रोसेसिंग**: आपकी तस्वीरें केवल अस्थायी रैम (RAM) में प्रोसेस होती हैं और डिस्क पर कभी सेव नहीं की जातीं।\n"
                    "- **शून्य व्यक्तिगत डेटा साझाकरण**: रोगी की पहचान या तस्वीरें किसी बाहरी क्लाउड एलएलएम को नहीं भेजी जाती हैं।\n"
                    "- **अनुसंधान प्रोटोटाइप**: यह एक नैदानिक अनुसंधान उपकरण है और प्रमाणित डॉक्टर की जांच का विकल्प नहीं है।"
                ),
                "speak_text": "आपकी तस्वीरें केवल रैम में प्रोसेस होती हैं और कभी डिस्क पर सुरक्षित नहीं रखी जातीं। आपकी गोपनीयता पूरी तरह सुरक्षित है।",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "सुरक्षित जांच जारी रखने के लिए अगला चरण चुनें।",
                "mode": "deterministic"
            }
        else:
            return {
                "message": (
                    "### Clinical Safety, Ethics & Privacy Standards\n\n"
                    "- **In-Memory RAM Processing**: Your eye captures are processed strictly in RAM and immediately discarded. No photos are retained on disk.\n"
                    "- **Zero PII Exposure**: Patient demographic information is stripped and never transmitted to external cloud LLM APIs.\n"
                    "- **Investigational Tool**: EyeHealthAI is a research screening prototype and does not replace in-person laboratory evaluation."
                ),
                "speak_text": "Your captures are processed strictly in RAM and never stored permanently, ensuring complete patient privacy.",
                "expression": "default",
                "severity": "info",
                "suggested_next_step": "Continue safely with your screening workflow.",
                "mode": "deterministic"
            }

    # Knowledge Base: Reports & PDF
    elif any(k in q_low for k in ["report", "pdf", "scorecard", "download", "results", "variance", "रिपोर्ट", "डाउनलोड"]):
        if clean_lang == "hi":
            return {
                "message": (
                    "### 2-पृष्ठों की नैदानिक पीडीएफ रिपोर्ट\n\n"
                    "- **पृष्ठ 1 (क्लिनिकल स्कोरकार्ड)**: अनुमानित हीमोग्लोबिन (g/dL), WHO एनीमिया श्रेणी, सत्र-स्तरीय भिन्नता, और गुणवत्ता ऑडिट।\n"
                    "- **पृष्ठ 2 (बायोमार्कर विश्लेषण)**: क्रोमैटिक कलर इंडेक्स (RGB/Lab), विश्वसनीयता गेट स्थिति, और डॉक्टर परामर्श मार्गदर्शन।"
                ),
                "speak_text": "जांच पूरी होने पर आप 2-पृष्ठों की विस्तृत रिपोर्ट डाउनलोड कर सकते हैं जिसमें स्कोरकार्ड और बायोमार्कर विश्लेषण शामिल है।",
                "expression": "happy",
                "severity": "info",
                "suggested_next_step": "जांच पूरी होने के बाद 'Download PDF Report' पर क्लिक करें।",
                "mode": "deterministic"
            }
        else:
            return {
                "message": (
                    "### 2-Page Clinical PDF Screening Report\n\n"
                    "- **Page 1 (Clinical Scorecard)**: Displays predicted hemoglobin (g/dL), WHO anemia severity classification, multi-view session variance, and optical quality verification.\n"
                    "- **Page 2 (Biomarker Analysis)**: Details chromatic color-space indices (RGB, CIE-Lab, Erythema Index), reliability gate status, and structured doctor consultation guidance."
                ),
                "speak_text": "You can download a comprehensive 2-page clinical PDF report summarizing your scorecard, WHO staging, chromatic biomarkers, and doctor guidance.",
                "expression": "happy",
                "severity": "info",
                "suggested_next_step": "Click 'Download PDF Report' on Step 7 after inference.",
                "mode": "deterministic"
            }

    # Friendly Intelligent Default Guide (Never a static Step 1 instruction)
    else:
        if clean_lang == "hi":
            return {
                "message": (
                    "नमस्ते! मैं एआई डॉ. एस्ट्रा हूँ, आपका नेत्र स्वास्थ्य अनुसंधान गाइड।\n\n"
                    "मैं EyeHealthAI परियोजना के बारे में किसी भी तकनीकी या नैदानिक प्रश्न का उत्तर दे सकती हूँ:\n"
                    "- 🩸 **एनीमिया मॉडल**: ResNet-50 हीमोग्लोबिन अनुमान (~1.50 g/dL MAE)\n"
                    "- 👁️ **कंजंक्टिवाइटिस मॉडल**: SLID स्लिट-लैंप क्लासिफायर (98.78% सटीकता)\n"
                    "- 🟡 **पीलिया मॉडल**: स्क्लेरा के पीलेपन की पहचान\n"
                    "- 📸 **कैमरा मार्गदर्शन**: सही रोशनी, दूरी (10-15 सेमी) और स्थिति\n"
                    "- 🔒 **सुरक्षा व गोपनीयता**: रैम-ओनली प्रोसेसिंग और शून्य डेटा लीकेज\n\n"
                    "आप इनमें से किसी भी विषय पर विस्तार से पूछ सकते हैं!"
                ),
                "speak_text": "नमस्ते! मैं डॉक्टर एस्ट्रा हूँ। आप एनीमिया, कंजंक्टिवाइटिस, पीलिया मॉडल, या कैमरा कैप्चर के बारे में कोई भी प्रश्न पूछ सकते हैं।",
                "expression": "happy",
                "severity": "info",
                "suggested_next_step": "किसी मॉडल या कैप्चर तकनीक के बारे में पूछें।",
                "mode": "deterministic"
            }
        else:
            return {
                "message": (
                    "Hello! I'm AI Dr. Astra, your AI Eye Health Guide & Research Companion.\n\n"
                    "I can provide detailed technical and clinical insights into any aspect of EyeHealthAI:\n"
                    "- 🩸 **Anemia Model**: ResNet-50 continuous Hb regression (~1.50 g/dL MAE, WHO staging)\n"
                    "- 👁️ **Conjunctivitis Model**: SLID slit-lamp injection classifier (98.78% accuracy, Mendeley shortcut gate)\n"
                    "- 🟡 **Jaundice Model**: Scleral icterus chromatic detection and ground-truth validation\n"
                    "- 📸 **Capture Guidance**: Proper lighting, macro distance (10–15 cm), and alignment\n"
                    "- 🔒 **Safety & Privacy**: RAM-only processing, zero PII transfer, and patient group splitting\n\n"
                    "Feel free to ask about any specific model, evaluation metric, or screening step!"
                ),
                "speak_text": "Hello! I am Doctor Astra. You can ask me about our anemia, conjunctivitis, or jaundice models, camera capture tips, or project architecture.",
                "expression": "happy",
                "severity": "info",
                "suggested_next_step": "Ask a question about any model, metric, or capture step.",
                "mode": "deterministic"
            }
