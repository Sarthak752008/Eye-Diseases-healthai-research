"""
create_notebooks.py - Creates the 3 model notebooks:
  - notebooks/02_anemia_model.ipynb
  - notebooks/03_conjunctivitis_model.ipynb
  - notebooks/04_jaundice_model.ipynb
"""

import os
import json

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def create_anemia_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Model A — Anemia Hemoglobin (Hb) Regression\n",
                "\n",
                "This notebook implements and evaluates the **Model A (ResNet-50)** regressor predicting numerical laboratory hemoglobin (Hb in g/dL) from non-invasive ocular images.\n",
                "\n",
                "### Clinical Constraints Enforced:\n",
                "1. **Never fabricate ground truth**: Strict pairing with laboratory Hb spreadsheets (`India.xlsx`, `Italy.xlsx`).\n",
                "2. **Strict Patient Leakage Prevention**: All multi-view images belonging to the same subject reside strictly in the same split (Train, Val, or Test).\n",
                "3. **Color-space preservation**: No aggressive color jitter that would distort conjunctival erythema/pallor."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import torch\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "\n",
                "# Add workspace root\n",
                "WORKSPACE_ROOT = os.path.abspath('..')\n",
                "if WORKSPACE_ROOT not in sys.path:\n",
                "    sys.path.insert(0, WORKSPACE_ROOT)\n",
                "\n",
                "from src.preprocessing.dataset_anemia import build_anemia_dataloaders\n",
                "from src.models.anemia_model import AnemiaRegressor, load_anemia_checkpoint\n",
                "from src.evaluation.evaluate import compute_regression_metrics\n",
                "\n",
                "print('Torch version:', torch.__version__)\n",
                "print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Processed Manifests and DataLoaders"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'anemia_manifest.csv')\n",
                "df = pd.read_csv(manifest_path)\n",
                "print(f'Total records in manifest: {len(df)}')\n",
                "print('Split distribution by subject:')\n",
                "print(df.groupby('split')['subject_id'].nunique())\n",
                "\n",
                "loaders = build_anemia_dataloaders(\n",
                "    manifest_csv=manifest_path,\n",
                "    workspace_root=WORKSPACE_ROOT,\n",
                "    batch_size=16,\n",
                "    use_processed_dir=os.path.join(WORKSPACE_ROOT, 'data', 'processed', 'anemia')\n",
                ")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load Trained ResNet50 Regressor & Evaluate"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "checkpoint_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'anemia', 'best_model.pth')\n",
                "if os.path.exists(checkpoint_path):\n",
                "    model = load_anemia_checkpoint(checkpoint_path)\n",
                "    print('Loaded checkpoint from:', checkpoint_path)\n",
                "else:\n",
                "    print('Checkpoint not found. Run scripts/train_anemia.py first.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Evaluate on test loader\n",
                "model.eval()\n",
                "all_preds, all_targets = [], []\n",
                "with torch.no_grad():\n",
                "    for batch in loaders['test']:\n",
                "        images = batch['image']\n",
                "        targets = batch['hb']\n",
                "        preds = model(images)\n",
                "        all_preds.extend(preds.numpy())\n",
                "        all_targets.extend(targets.numpy())\n",
                "\n",
                "metrics = compute_regression_metrics(np.array(all_targets), np.array(all_preds))\n",
                "print('=== Test Evaluation Metrics ===')\n",
                "print(f'MAE:  {metrics[\"mae_g_dl\"]} g/dL')\n",
                "print(f'RMSE: {metrics[\"rmse_g_dl\"]} g/dL')\n",
                "print(f'R²:   {metrics[\"r2\"]}')\n",
                "print(f'Pearson r: {metrics[\"pearson_r\"]}')"
            ]
        }
    ]
    return make_notebook(cells)


def create_conjunctivitis_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Model B — Conjunctivitis Binary Classification\n",
                "\n",
                "This notebook implements and evaluates the **Model B (MobileNetV3-Small)** classifier differentiating Normal ocular surface versus Conjunctivitis.\n",
                "\n",
                "### Clinical Constraints Enforced:\n",
                "1. **Never interpret general redness as proof of infectious conjunctivitis**: Differentiates Normal vs. Conjunctivitis using validated multi-disease eye dataset.\n",
                "2. **Independent model head**: Separate architecture and cross-entropy loss with class reweighting.\n",
                "3. **Preserve clinical colors**: No solarization or color inversion."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import torch\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "\n",
                "WORKSPACE_ROOT = os.path.abspath('..')\n",
                "if WORKSPACE_ROOT not in sys.path:\n",
                "    sys.path.insert(0, WORKSPACE_ROOT)\n",
                "\n",
                "from src.preprocessing.dataset_conjunctivitis import build_conjunctivitis_dataloaders\n",
                "from src.models.conjunctivitis_model import load_conjunctivitis_checkpoint\n",
                "from src.evaluation.evaluate import compute_classification_metrics\n",
                "\n",
                "print('Torch version:', torch.__version__)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Processed Manifests & DataLoaders"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "manifest_path = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'conjunctivitis_manifest.csv')\n",
                "df = pd.read_csv(manifest_path)\n",
                "print(f'Total records in manifest: {len(df)}')\n",
                "print('Class distribution:', df['class_name'].value_counts().to_dict())\n",
                "\n",
                "loaders = build_conjunctivitis_dataloaders(\n",
                "    manifest_csv=manifest_path,\n",
                "    workspace_root=WORKSPACE_ROOT,\n",
                "    batch_size=32,\n",
                "    use_processed_dir=os.path.join(WORKSPACE_ROOT, 'data', 'processed', 'conjunctivitis')\n",
                ")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load Checkpoint and Evaluate"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "checkpoint_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'conjunctivitis', 'best_model.pth')\n",
                "if os.path.exists(checkpoint_path):\n",
                "    model = load_conjunctivitis_checkpoint(checkpoint_path)\n",
                "    print('Loaded checkpoint:', checkpoint_path)\n",
                "else:\n",
                "    print('Checkpoint not found. Run scripts/train_conjunctivitis.py first.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "model.eval()\n",
                "all_preds, all_probs, all_labels = [], [], []\n",
                "with torch.no_grad():\n",
                "    for batch in loaders['test']:\n",
                "        images = batch['image']\n",
                "        labels = batch['label']\n",
                "        outputs = model(images)\n",
                "        probs = torch.softmax(outputs, dim=1)[:, 1]\n",
                "        preds = torch.argmax(outputs, dim=1)\n",
                "        all_preds.extend(preds.numpy())\n",
                "        all_probs.extend(probs.numpy())\n",
                "        all_labels.extend(labels.numpy())\n",
                "\n",
                "metrics = compute_classification_metrics(all_labels, all_preds, all_probs)\n",
                "print('=== Test Evaluation Metrics ===')\n",
                "print(f'Accuracy:  {metrics[\"accuracy\"]:.4f}')\n",
                "print(f'Precision: {metrics[\"precision\"]:.4f}')\n",
                "print(f'Recall:    {metrics[\"recall_sensitivity\"]:.4f}')\n",
                "print(f'F1-Score:  {metrics[\"f1_score\"]:.4f}')\n",
                "print(f'ROC-AUC:   {metrics[\"roc_auc\"]:.4f}')\n",
                "print(f'Confusion Matrix:\\n', np.array(metrics['confusion_matrix']))"
            ]
        }
    ]
    return make_notebook(cells)


def create_jaundice_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Model C — Jaundice (Scleral Icterus) Pipeline & Audit Status\n",
                "\n",
                "This notebook documents the **Model C (MobileNetV3-Small)** architecture, color-preservation protocols, and audit findings.\n",
                "\n",
                "### Audit Status: BLOCKER (Dataset Not In Workspace)\n",
                "- **Clinical Safety Protocol**: Never fabricate synthetic bilirubin values or invent labels.\n",
                "- **Readiness**: Reusable pipeline (`src/preprocessing/dataset_jaundice.py`) and model architecture (`src/models/jaundice_model.py`) are fully implemented and unit-tested to run as soon as clinical data arrives."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import torch\n",
                "\n",
                "WORKSPACE_ROOT = os.path.abspath('..')\n",
                "if WORKSPACE_ROOT not in sys.path:\n",
                "    sys.path.insert(0, WORKSPACE_ROOT)\n",
                "\n",
                "from src.models.jaundice_model import JaundiceClassifier\n",
                "\n",
                "status_path = os.path.join(WORKSPACE_ROOT, 'saved_models', 'jaundice', 'jaundice_config.json')\n",
                "with open(status_path, 'r') as f:\n",
                "    config = json.load(f)\n",
                "print('Jaundice Pipeline Configuration:')\n",
                "print(json.dumps(config, indent=2))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Architecture Verification on Forward Pass"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Verify architecture\n",
                "model = JaundiceClassifier(pretrained=True, num_classes=2)\n",
                "dummy_batch = torch.randn(2, 3, 224, 224)\n",
                "logits = model(dummy_batch)\n",
                "print('Output logits shape:', logits.shape)\n",
                "print('Architecture is verified and ready for dataset ingestion.')"
            ]
        }
    ]
    return make_notebook(cells)


def main():
    nb_anemia = os.path.join(NOTEBOOKS_DIR, '02_anemia_model.ipynb')
    with open(nb_anemia, 'w', encoding='utf-8') as f:
        json.dump(create_anemia_notebook(), f, indent=2)
    print(f"Created {nb_anemia}")

    nb_conj = os.path.join(NOTEBOOKS_DIR, '03_conjunctivitis_model.ipynb')
    with open(nb_conj, 'w', encoding='utf-8') as f:
        json.dump(create_conjunctivitis_notebook(), f, indent=2)
    print(f"Created {nb_conj}")

    nb_jaund = os.path.join(NOTEBOOKS_DIR, '04_jaundice_model.ipynb')
    with open(nb_jaund, 'w', encoding='utf-8') as f:
        json.dump(create_jaundice_notebook(), f, indent=2)
    print(f"Created {nb_jaund}")


if __name__ == '__main__':
    main()
