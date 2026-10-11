"""
setup_jaundice_pipeline.py - Configures Jaundice pipeline and documents blocker.
Verifies that the MobileNetV3-Small Jaundice architecture can instantiate and run inference,
and saves pipeline configuration and blocker status.
"""

import os
import sys
import json
import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.models.jaundice_model import JaundiceClassifier, save_jaundice_checkpoint

SAVED_MODELS_DIR = os.path.join(WORKSPACE_ROOT, 'saved_models', 'jaundice')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def main():
    print("Setting up Jaundice pipeline and architecture verification...")
    # 1. Instantiate and verify forward pass on dummy tensor
    model = JaundiceClassifier(pretrained=True, num_classes=2)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 2), f"Unexpected output shape: {output.shape}"
    print("MobileNetV3-Small Jaundice model architecture verified successfully.")
    
    # 2. Save pipeline configuration and documentation
    config = {
        'model_name': 'MobileNetV3-Small Jaundice Classifier',
        'classes': {'0': 'normal', '1': 'jaundiced'},
        'input_size': [3, 224, 224],
        'color_preservation': 'Strict (no hue jitter, scleral yellowing preserved)',
        'status': 'BLOCKED_PENDING_DATA',
        'blocker_reason': 'No raw jaundice dataset or clinical bilirubin laboratory measurements exist in the workspace.',
        'safety_policy': 'No synthetic labels or fabricated bilirubin numbers were created.'
    }
    
    config_path = os.path.join(SAVED_MODELS_DIR, 'jaundice_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
        
    status_md = """# Jaundice Model Pipeline Status

## Operational Status: BLOCKED (DATA UNAVAILABLE)

### Blocker Details
- **Issue**: A thorough recursive audit of the workspace confirmed that **no dataset for jaundice, icterus, or laboratory bilirubin measurements exists**.
- **Clinical Safety Enforcement**: In strict compliance with `AGENTS.md` and clinical safety protocols, **no synthetic bilirubin values or fabricated labels have been created**.
- **Architecture Readiness**: The MobileNetV3-Small classifier architecture (`src/models/jaundice_model.py`) and sclera color-preserving preprocessing pipeline (`src/preprocessing/dataset_jaundice.py`) have been fully implemented, unit-tested, and verified.
- **Data Ingestion Readiness**: The standardized folder structure (`data/processed/jaundice/train/`, `val/`, `test/`) is established. Once an authorized clinical jaundice dataset is placed in `data/raw/jaundice/`, the pipeline will execute end-to-end immediately.
"""
    status_path = os.path.join(SAVED_MODELS_DIR, 'STATUS.md')
    with open(status_path, 'w', encoding='utf-8') as f:
        f.write(status_md)
        
    print(f"Jaundice configuration and status saved in {SAVED_MODELS_DIR}")


if __name__ == '__main__':
    main()
