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

    # Step 3: Try Gemini API with candidate fallback chain
    client = _get_gemini_client()
    candidate_models = [
        getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"),
        "gemini-flash-latest",
        "gemini-3.8-flash",
        "gemini-2.5-flash-lite",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
    ]

    if client:
        system_prompt = (
            f"{_SYSTEM_INSTRUCTION}\n\n"
            f"You are AI Dr. Astra, an expert ocular health guide and research companion. "
            f"The user is at Step {current_step} of the screening workflow. "
            f"Respond in {'Hindi (Devanagari script)' if clean_lang == 'hi' else 'English'}. "
            f"Keep answers clear, highly informative, structured with markdown, and strictly within safety boundaries."
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
                logger.debug(f"Gemini model {target_model} candidate failed: {e}")

    # Step 4: Comprehensive Clinical & Technical Knowledge Engine (Deterministic Fallback)
    q_low = sanitized_q.lower()

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
