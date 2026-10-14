"""
src/inference/copilot.py
------------------------
Gemini Research Copilot Service.

Provides an optional LLM-powered research assistant for:
  - Research paper Q&A
  - Dataset structure & manifest insights
  - Model architecture & metric interpretation
  - Code debugging guidance
  - Documentation lookup

SAFETY GUARDRAILS (MANDATORY):
  1. NEVER send patient images or binary data to the LLM.
  2. NEVER forward raw clinical results (Hb values, bilirubin, diagnostic outputs).
  3. NEVER include filesystem paths that could reveal patient identity.
  4. API key is loaded server-side from .env — never exposed to the frontend.
"""

import os
import re
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII & Medical Data Sanitization
# ---------------------------------------------------------------------------

# Patterns that MUST be stripped before sending to LLM
_SANITIZE_PATTERNS = [
    # File system paths (Windows & Unix)
    (re.compile(r'[A-Za-z]:\\(?:[^\s\\]+\\)*[^\s\\]+', re.IGNORECASE), '[REDACTED_PATH]'),
    (re.compile(r'/(?:home|Users|mnt|data|datasets?|images?)/[^\s]+', re.IGNORECASE), '[REDACTED_PATH]'),
    # Hemoglobin values with units
    (re.compile(r'\b\d{1,2}\.?\d*\s*g\s*/?\s*d[Ll]\b'), '[REDACTED_HB_VALUE]'),
    # Bilirubin values
    (re.compile(r'\b\d{1,2}\.?\d*\s*(?:mg\s*/?\s*d[Ll]|µmol\s*/?\s*[Ll])\b'), '[REDACTED_BILIRUBIN_VALUE]'),
    # Phone numbers (Indian & international)
    (re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), '[REDACTED_PHONE]'),
    # Email addresses
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[REDACTED_EMAIL]'),
    # Base64 image data
    (re.compile(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{20,}'), '[REDACTED_IMAGE_DATA]'),
    # Patient ID patterns
    (re.compile(r'\bpatient[_\s-]?(?:id|ID)\s*[:=]\s*\S+', re.IGNORECASE), '[REDACTED_PATIENT_ID]'),
]

# Topics that are explicitly blocked
_BLOCKED_TOPIC_PATTERNS = [
    re.compile(r'\b(?:diagnos|prescri|treat|medic(?:at|in))\w*\b.*\bpatient\b', re.IGNORECASE),
    re.compile(r'\b(?:what\s+(?:is|are)\s+(?:my|this)\s+(?:disease|condition|diagnosis))\b', re.IGNORECASE),
    re.compile(r'\bshould\s+(?:I|they|the\s+patient)\s+(?:take|use|stop)\b', re.IGNORECASE),
]

_BLOCKED_RESPONSE = (
    "⚠️ I cannot provide medical diagnoses, treatment recommendations, or "
    "interpret clinical results for individual patients. I'm a research assistant "
    "focused on helping with dataset analysis, model architecture, research papers, "
    "and code debugging. Please consult a qualified healthcare professional for "
    "medical advice."
)


def sanitize_input(text: str) -> str:
    """
    Strips PII, clinical values, file paths, and image data from user input
    before forwarding to the Gemini API.
    """
    sanitized = text
    for pattern, replacement in _SANITIZE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def is_blocked_topic(text: str) -> bool:
    """Returns True if the query attempts to solicit medical advice."""
    for pattern in _BLOCKED_TOPIC_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Gemini API Client
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """You are the EyeHealthAI Research Copilot — an AI assistant embedded in a clinical computer vision research dashboard.

Your expertise covers:
- Ocular disease screening research (anemia from conjunctival pallor, conjunctivitis classification, jaundice/scleral icterus)
- CNN architectures (ResNet-50, MobileNetV3) for medical image analysis
- Dataset auditing, manifest integrity, and preprocessing pipelines
- Model evaluation metrics (MAE, RMSE, accuracy, precision, recall, F1)
- Data leakage prevention and patient-level train/val/test splitting
- Python/PyTorch/FastAPI debugging and code explanations
- Research paper summaries and methodology comparisons

STRICT RULES:
1. You are a RESEARCH assistant. NEVER provide medical diagnoses, treatment plans, or clinical advice.
2. NEVER interpret clinical results for individual patients.
3. If asked about patient-specific outcomes, decline and recommend consulting a healthcare professional.
4. Keep responses concise, technical, and evidence-based.
5. When discussing model performance, always include appropriate caveats about experimental/research status.
6. Format responses using markdown for readability."""

# Lazy singleton client
_gemini_client = None
_gemini_available = None


def _get_gemini_client():
    """Lazily initializes the Gemini client. Returns None if unavailable."""
    global _gemini_client, _gemini_available

    if _gemini_available is False:
        return None
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — Research Copilot will use offline mode.")
        _gemini_available = False
        return None

    try:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
        _gemini_available = True
        logger.info("Gemini Research Copilot initialized successfully.")
        return _gemini_client
    except ImportError:
        logger.warning("google-genai package not installed. Run: pip install google-genai")
        _gemini_available = False
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client: {e}")
        _gemini_available = False
        return None


def get_copilot_status() -> Dict[str, Any]:
    """Returns current copilot availability status."""
    client = _get_gemini_client()
    return {
        "available": client is not None,
        "mode": "gemini_api" if client else "offline",
        "model": "gemini-2.0-flash" if client else None,
        "safety_gates": [
            "PII/path sanitization active",
            "Clinical result redaction active",
            "Medical advice blocking active",
            "No image/binary data forwarding"
        ]
    }


# ---------------------------------------------------------------------------
# Offline Fallback Responses
# ---------------------------------------------------------------------------

_OFFLINE_RESPONSES = {
    "architecture": (
        "## EyeHealthAI Model Architecture\n\n"
        "| Module | Architecture | Task | Status |\n"
        "|--------|-------------|------|--------|\n"
        "| **Anemia** | ResNet-50 (pretrained) | Hb regression (g/dL) | ✅ Available (Research) |\n"
        "| **Conjunctivitis** | MobileNetV3-Small | Binary classification | 🚫 Blocked (Resolution Confound) |\n"
        "| **Jaundice** | MobileNetV3-Small | Binary classification | ⚠️ Unavailable (No Dataset) |\n\n"
        "The anemia model uses a pretrained ResNet-50 backbone with a custom regression head. "
        "It predicts continuous hemoglobin values from conjunctival pallor images."
    ),
    "metrics": (
        "## Model Evaluation Metrics\n\n"
        "**Anemia Regressor (ResNet-50):**\n"
        "- Test MAE: ~1.50 g/dL\n"
        "- Evaluated on patient-level grouped splits to prevent data leakage\n\n"
        "**Conjunctivitis Classifier (MobileNetV3-Small):**\n"
        "- ⚠️ Achieved 100% accuracy — but this is due to a resolution/framing shortcut\n"
        "- Normal images: ~56×56 px squares; Conjunctivitis: ~352×252 px rectangles\n"
        "- A metadata-only classifier achieves identical results → predictions withheld\n\n"
        "**Jaundice:** No model trained yet — no clinical dataset available."
    ),
    "dataset": (
        "## Dataset Structure\n\n"
        "**Anemia Dataset:**\n"
        "- Source: India.xlsx + Italy.xlsx clinical reference spreadsheets\n"
        "- Images paired with laboratory hemoglobin measurements (g/dL)\n"
        "- Patient/session-level grouping enforced during splitting\n\n"
        "**Conjunctivitis Dataset:**\n"
        "- Binary: Normal vs. Conjunctivitis\n"
        "- ⚠️ Systematic resolution confound detected between classes\n\n"
        "**Jaundice:** No dataset exists in the workspace."
    ),
    "default": (
        "I'm currently in **offline mode** (Gemini API unavailable). "
        "I can still help with pre-loaded knowledge about:\n\n"
        "- 🏗️ **Architecture** — Model designs and layer configurations\n"
        "- 📊 **Metrics** — Evaluation results and performance benchmarks\n"
        "- 📁 **Dataset** — Data structure, manifests, and splitting strategies\n"
        "- 🔒 **Safety** — Reliability gates and clinical guardrails\n\n"
        "Try asking about one of these topics, or check that `GEMINI_API_KEY` is set in your `.env` file."
    ),
}


def _get_offline_response(query: str) -> str:
    """Keyword-match fallback when Gemini API is unavailable."""
    q = query.lower()
    if any(k in q for k in ('architect', 'resnet', 'mobilenet', 'model design', 'backbone', 'cnn')):
        return _OFFLINE_RESPONSES["architecture"]
    if any(k in q for k in ('metric', 'mae', 'rmse', 'accuracy', 'f1', 'performance', 'evaluation')):
        return _OFFLINE_RESPONSES["metrics"]
    if any(k in q for k in ('dataset', 'manifest', 'split', 'india.xlsx', 'italy.xlsx', 'preprocess')):
        return _OFFLINE_RESPONSES["dataset"]
    return _OFFLINE_RESPONSES["default"]


# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------

async def chat(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Process a research copilot chat message.

    Args:
        message: The user's question/message.
        conversation_history: Optional list of prior messages for context.

    Returns:
        Dict with 'response', 'mode', and 'sanitized' fields.
    """
    if not message or not message.strip():
        return {
            "response": "Please enter a question about the EyeHealthAI research project.",
            "mode": "error",
            "sanitized": False
        }

    # Step 1: Check for blocked medical-advice topics
    if is_blocked_topic(message):
        return {
            "response": _BLOCKED_RESPONSE,
            "mode": "blocked",
            "sanitized": False
        }

    # Step 2: Sanitize the input (strip PII, paths, clinical values)
    sanitized_message = sanitize_input(message)
    was_sanitized = (sanitized_message != message)

    # Step 3: Try Gemini API
    client = _get_gemini_client()
    if client is None:
        # Offline fallback
        offline_response = _get_offline_response(message)
        return {
            "response": offline_response,
            "mode": "offline",
            "sanitized": was_sanitized
        }

    # Step 4: Build conversation contents for Gemini
    try:
        from src.config import settings
        candidate_models = [
            getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"),
            "gemini-flash-latest",
            "gemini-3.8-flash",
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
        ]
        contents = []

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-6:]:  # Keep last 6 messages for context
                role = msg.get("role", "user")
                text = msg.get("content", "")
                if role == "user":
                    text = sanitize_input(text)
                contents.append({
                    "role": role,
                    "parts": [{"text": text}]
                })

        # Add current message
        contents.append({
            "role": "user",
            "parts": [{"text": sanitized_message}]
        })

        response = None
        for target_model in candidate_models:
            try:
                response = client.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config={
                        "system_instruction": _SYSTEM_INSTRUCTION,
                        "temperature": 0.3,
                        "max_output_tokens": 1024,
                    }
                )
                if response and response.text:
                    break
            except Exception as e:
                logger.debug(f"Copilot model candidate {target_model} failed: {e}")

        if response and response.text:
            return {
                "response": response.text.strip(),
                "mode": "gemini_api",
                "sanitized": was_sanitized
            }

        # If all candidates failed, gracefully degrade to rich offline knowledge
        offline_response = _get_offline_response(message)
        return {
            "response": offline_response,
            "mode": "offline",
            "sanitized": was_sanitized
        }

    except Exception as e:
        error_str = str(e)
        logger.error(f"Gemini API error: {error_str}")

        # Graceful degradation to offline mode
        if "403" in error_str or "PERMISSION_DENIED" in error_str:
            offline_resp = _get_offline_response(message)
            return {
                "response": (
                    "⚠️ **Gemini API returned a permission error.** "
                    "The API key may need additional permissions or the model may not be available in your region.\n\n"
                    "---\n\n" + offline_resp
                ),
                "mode": "offline_fallback",
                "sanitized": was_sanitized
            }

        offline_resp = _get_offline_response(message)
        return {
            "response": (
                f"⚠️ **Gemini API error:** Could not reach the language model.\n\n"
                "---\n\n" + offline_resp
            ),
            "mode": "offline_fallback",
            "sanitized": was_sanitized
        }
