"""
Models module for EyeHealthAI
"""

from .anemia_model import AnemiaRegressor, save_anemia_checkpoint, load_anemia_checkpoint
from .conjunctivitis_model import (
    ConjunctivitisClassifier,
    save_conjunctivitis_checkpoint,
    load_conjunctivitis_checkpoint
)
from .jaundice_model import (
    JaundiceClassifier,
    JaundiceRegressor,
    save_jaundice_checkpoint,
    load_jaundice_checkpoint
)
