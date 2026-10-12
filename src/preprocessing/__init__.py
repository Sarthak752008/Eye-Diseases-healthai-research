"""
preprocessing module for EyeHealthAI
"""

from .image_quality import safe_load_image, calculate_image_quality, resize_and_pad

try:
    from .dataset_anemia import AnemiaDataset, build_anemia_dataloaders, get_anemia_transforms
    from .dataset_conjunctivitis import ConjunctivitisDataset, build_conjunctivitis_dataloaders, get_conjunctivitis_transforms
    from .dataset_jaundice import JaundiceDataset, build_jaundice_dataloaders, get_jaundice_transforms
except (ImportError, Exception):
    pass
