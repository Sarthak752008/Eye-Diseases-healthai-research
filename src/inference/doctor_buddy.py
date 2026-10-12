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
    history: Optional[List[dict]] = None
) -> Dict[str, Any]:
    """
    Processes a user-asked question through Gemini LLM if available,
    with complete PII/medical-advice safety filtering and deterministic fallback.
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

    # Step 3: Try Gemini API
    client = _get_gemini_client()
    target_model = getattr(settings, "GEMINI_MODEL", "gemini-2.0-flash")

    if client:
        try:
            system_prompt = (
                f"{_SYSTEM_INSTRUCTION}\n\n"
                f"You are AI Dr. Astra, a friendly, concise ocular health guide and research companion. "
                f"The user is at Step {current_step} of the screening workflow. "
                f"Respond in {'Hindi (Devanagari script)' if clean_lang == 'hi' else 'English'}. "
                f"Keep answers brief (under 3 sentences), warm, encouraging, and strictly within safety boundaries."
            )
            # Format sanitized history if provided
            contents = []
            if history and isinstance(history, list):
                for h in history[-4:]:
                    role = "user" if h.get("role") == "user" else "model"
                    h_text = sanitize_input(str(h.get("content", "")))
                    if h_text:
                        contents.append({"role": role, "parts": [{"text": h_text}]})
            contents.append({"role": "user", "parts": [{"text": sanitized_q}]})

            response = client.models.generate_content(
                model=target_model,
                contents=contents,
                config={
                    "system_instruction": system_prompt,
                    "temperature": 0.3,
                    "max_output_tokens": 256,
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
            logger.warning(f"Doctor Buddy Gemini call failed: {e}. Using deterministic guidance.")

    # Step 4: Deterministic keyword fallback
    q_low = sanitized_q.lower()
    if any(k in q_low for k in ["camera", "permission", "allow", "webcam", "कैमरा", "अनुमति"]):
        return get_doctor_buddy_guidance("camera_permission_prompt", clean_lang)
    elif any(k in q_low for k in ["blur", "focus", "sharp", "धुंधला", "साफ"]):
        return get_doctor_buddy_guidance("quality_blurry", clean_lang)
    elif any(k in q_low for k in ["light", "dark", "bright", "रोशनी", "अंधेरा"]):
        return get_doctor_buddy_guidance("quality_dark", clean_lang)
    elif any(k in q_low for k in ["anemia", "hemoglobin", "hb", "एनीमिया", "हीमोग्लोबिन"]):
        return {
            "message": (
                "The Anemia module estimates hemoglobin from the conjunctival pallor of the lower inner eyelid. "
                "It is a research tool evaluated with an MAE of ~1.50 g/dL. Laboratory CBC tests are always required for clinical diagnosis."
                if clean_lang == "en" else
                "एनीमिया मॉड्यूल निचली पलक के अंदरूनी हिस्से से हीमोग्लोबिन का अनुमान लगाता है। "
                "यह एक शोध उपकरण है (~1.50 g/dL MAE)। चिकित्सीय पुष्टि हेतु सीबीसी रक्त परीक्षण अनिवार्य है।"
            ),
            "speak_text": "The anemia module estimates hemoglobin from eyelid pallor for research." if clean_lang == "en" else "एनीमिया मॉड्यूल पलक के रंग से हीमोग्लोबिन का अनुमान लगाता है।",
            "expression": "default",
            "severity": "info",
            "suggested_next_step": "Follow the capture guide to expose the lower eyelid.",
            "mode": "deterministic"
        }
    elif any(k in q_low for k in ["jaundice", "yellow", "bilirubin", "पीलिया", "पीला"]):
        return {
            "message": (
                "The Jaundice module screens for visible scleral yellowing (icterus). "
                "Numerical serum bilirubin estimation is unavailable because paired laboratory bilirubin datasets do not exist."
                if clean_lang == "en" else
                "पीलिया मॉड्यूल आंख के सफेद हिस्से में पीलेपन की जांच करता है। "
                "संख्यात्मक बिलीरुबिन मान उपलब्ध नहीं है क्योंकि प्रयोगशाला डेटासेट मौजूद नहीं है।"
            ),
            "speak_text": "The jaundice module screens for visible scleral yellowing." if clean_lang == "en" else "पीलिया मॉड्यूल आंख के सफेद हिस्से में पीलेपन की जांच करता है।",
            "expression": "default",
            "severity": "info",
            "suggested_next_step": "Select the Jaundice module for screening.",
            "mode": "deterministic"
        }
    elif any(k in q_low for k in ["conjunctivitis", "pink eye", "red", "कंजंक्टिवाइटिस", "लाल"]):
        return get_doctor_buddy_guidance("model_mendeley_blocked", clean_lang)
    else:
        return get_doctor_buddy_guidance("welcome", clean_lang)
