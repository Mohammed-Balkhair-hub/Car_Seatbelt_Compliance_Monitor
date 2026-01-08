"""Seatbelt Detection Module"""

import cv2
import os
from pathlib import Path
from typing import Optional


def detect_seatbelt(
    car_image_path: str,
    model,
    conf: float = 0.55,
    ann_folder: Optional[str] = None,
    save_name: Optional[str] = None
) -> bool:
    """
    Detect seatbelt in a cropped car image.
    
    Detects seatbelt-related objects:
    - Class 9: person-noseatbelt
    - Class 10: person-seatbelt
    - Class 11: seatbelt
    
    Args:
        car_image_path: Path to the cropped car image
        model: YOLO model instance for seatbelt detection
        conf: Confidence threshold for seatbelt detection (default: 0.55)
        ann_folder: Directory to save annotated images (optional)
        save_name: Base name for saved files (optional, will use image filename if not provided)
    
    Returns:
        bool: True if seatbelt detected, False otherwise
    """
    # Get image name for saving if not provided
    if save_name is None:
        save_name = Path(car_image_path).stem
    
    # Read image
    img = cv2.imread(car_image_path)
    if img is None:
        raise ValueError(f"Could not read image: {car_image_path}")
    
    # Predict seatbelts (classes 9, 10, 11)
    results = model.predict(
        source=car_image_path,
        classes=[9, 10, 11],
        conf=conf,
        verbose=False
    )
    
    seatbelt_detected = False
    
    for r in results:
        # Save annotated image if folder specified
        if ann_folder is not None:
            annotated_image = r.plot()
            os.makedirs(ann_folder, exist_ok=True)
            cv2.imwrite(f'{ann_folder}/{save_name}_Annotated.jpg', annotated_image)
        
        # Check if any seatbelt-related objects detected
        if len(r.boxes) > 0:
            seatbelt_detected = True
    
    return seatbelt_detected
