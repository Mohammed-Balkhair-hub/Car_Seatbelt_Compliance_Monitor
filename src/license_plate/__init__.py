"""License Plate Detection Module"""

from .detector import detect_and_crop_plates
from .text_extractor import extract_plate_text

__all__ = ['detect_and_crop_plates', 'extract_plate_text']
