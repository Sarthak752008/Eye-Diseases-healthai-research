import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import torch
from PIL import Image
from src.config import WORKSPACE_ROOT
from src.models.anemia_model import load_anemia_checkpoint
from src.models.conjunctivitis_model import load_conjunctivitis_checkpoint
from src.models.jaundice_model import load_jaundice_checkpoint
from src.inference.attribution_cam import generate_model_attribution_package

def test_attribution():
    img_pil = Image.new("RGB", (224, 224), color=(180, 80, 80))
    tensor = torch.randn(1, 3, 224, 224)

    # 1. Anemia ResNet50
    ma = load_anemia_checkpoint(os.path.join(WORKSPACE_ROOT, 'saved_models', 'anemia', 'best_model.pth'))
    pkg_a = generate_model_attribution_package(
        model=ma,
        target_layer=ma.backbone.layer4[-1],
        input_tensor=tensor,
        final_input_pil=img_pil,
        is_regression=True,
        model_name="ResNet-50",
        target_name="Hemoglobin / Anemia"
    )
    print("Anemia GradCAM status:", pkg_a["status"], "has_overlay:", bool(pkg_a.get("overlay_b64")))

    # 2. SLID MobileNetV3
    ms = load_conjunctivitis_checkpoint(os.path.join(WORKSPACE_ROOT, 'saved_models', 'slid', 'best_model.pth'))
    pkg_s = generate_model_attribution_package(
        model=ms,
        target_layer=ms.backbone.features[-1],
        input_tensor=tensor,
        final_input_pil=img_pil,
        target_class=1,
        is_regression=False,
        model_name="MobileNetV3-Small",
        target_name="SLID Hyperemia"
    )
    print("SLID GradCAM status:", pkg_s["status"], "has_overlay:", bool(pkg_s.get("overlay_b64")))

    # 3. Jaundice MobileNetV3
    mj = load_jaundice_checkpoint(os.path.join(WORKSPACE_ROOT, 'saved_models', 'jaundice', 'best_model.pth'))
    pkg_j = generate_model_attribution_package(
        model=mj,
        target_layer=mj.backbone.features[-1],
        input_tensor=tensor,
        final_input_pil=img_pil,
        target_class=1,
        is_regression=False,
        model_name="MobileNetV3-Small",
        target_name="Scleral Icterus"
    )
    print("Jaundice GradCAM status:", pkg_j["status"], "has_overlay:", bool(pkg_j.get("overlay_b64")))

if __name__ == "__main__":
    test_attribution()
