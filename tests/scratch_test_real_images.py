import os
import sys
import json
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# Add workspace to sys.path
sys.path.insert(0, r"c:\Users\sarth\OneDrive\Pictures\Innotech")

from src.preprocessing.advanced_pipeline import process_ocular_image_advanced
from src.inference.unified_service import EyeHealthInferenceService

service = EyeHealthInferenceService()
folder = r"C:\Users\sarth\.gemini\antigravity\brain\70fdcf38-6d56-4d81-939c-67094dd39db0\.user_uploaded"
images = [
    "media_1791982296276_de397f8c.jpg",
    "media_1791982402907_3b6e56d5.jpg",
    "media_1791982402935_e0e7775e.jpg",
    "media_1791982472261_5c8f69a1.webp"
]

for name in images:
    p = os.path.join(folder, name)
    if not os.path.exists(p):
        print(f"File not found: {p}")
        continue
    im = Image.open(p)
    print("=" * 70)
    print(f"IMAGE: {name}")
    print(f"Original Dimensions: {im.size} | Mode: {im.mode} | Format: {im.format}")

    # Process through advanced pipeline
    res = process_ocular_image_advanced(im, generate_previews=True)
    det = res["detection"]
    enh = res["enhancement"]
    qg = res["quality_gate"]

    det_mode = det.get("detection_mode") or det.get("mode")
    crop_applied = det.get("crop_applied")
    crop_box = det.get("crop_box") or det.get("box")

    print(f"Detection Mode: {det_mode}")
    print(f"Eye Side: {det.get('eye_side')}")
    print(f"Crop Applied: {crop_applied} | Box: {crop_box}")
    print(f"Original vs Crop Dimensions: {im.size} -> {res['processed_image'].size}")
    print(f"Enhancement: applied={enh.get('applied')} | steps={enh.get('steps')}")
    print(f"Chroma Preserved: {enh.get('chromatic_channels_preserved')} | Delta: {enh.get('chroma_delta_mean')}")
    print(f"Quality Gate: is_suitable={qg.get('is_suitable')}")
    print(f"Metrics: Sharpness={qg.get('metrics', {}).get('sharpness_laplacian'):.1f}, Luminance={qg.get('metrics', {}).get('mean_luminance'):.1f}, Contrast={qg.get('metrics', {}).get('contrast_std'):.1f}")
    print(f"Quality Flags: {qg.get('flags')}")
    print(f"Model Suitability: {qg.get('model_suitability')}")
    print(f"Retake Guidance: {qg.get('retake_guidance_en')}")

    # Run unified inference
    inf = service.predict_unified([im], capture_source="upload", age_years=35, gender="male", external_ai_consent=False)
    anemia = inf.get("anemia") or {}
    slid = inf.get("slid") or {}
    jaund = inf.get("jaundice") or {}

    print("\n--- MODEL INFERENCE OUTPUTS ---")
    print(f"Model A (Anemia): status={anemia.get('status')} | Hb Pred={anemia.get('view_mean')} g/dL | Details: {anemia.get('prediction')}")
    print(f"Model B (SLID): status={slid.get('status')} | Details: {slid.get('prediction')}")
    print(f"Model C (Jaundice): status={jaund.get('status')} | Details: {jaund.get('prediction')}")
    print(f"Calibrated Fusion: {inf.get('calibrated_ensemble')}")
    print(f"Personalized AI Summary: {inf.get('personalized_ai', {}).get('bilingual', {}).get('en', {}).get('overall_summary')}")
