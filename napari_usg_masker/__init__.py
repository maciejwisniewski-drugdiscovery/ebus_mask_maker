"""
napari-usg-masker: A napari plugin for USG diagnostic video masking
"""

__version__ = "0.1.0"

from ._widget import USGMaskerWidget
from ._reader import napari_get_reader

__all__ = [
    "USGMaskerWidget", 
    "napari_get_reader"
]