"""License Plate Detection Module"""

import cv2
import os
from pathlib import Path
from typing import List, Optional


def detect_and_crop_plates(
    car_image_path: str,
    model,
    conf: float = 0.4,
    ann_folder: Optional[str] = None,
    crop_folder: Optional[str] = None,
    save_name: Optional[str] = None
) -> List[str]:
    """
    Detect and crop license plates from a car image using YOLO model.
    
    Args:
        car_image_path: Path to the car image
        model: YOLO model instance for license plate detection
        conf: Confidence threshold for plate detection (default: 0.4)
        ann_folder: Directory to save annotated images (optional)
        crop_folder: Directory to save cropped plate images (optional)
        save_name: Base name for saved files (optional, will use image filename if not provided)
    
    Returns:
        List of paths to cropped plate images, or empty list if no plates detected.
    """
    # Get image name for saving if not provided
    if save_name is None:
        save_name = Path(car_image_path).stem
    
    # Read image
    img = cv2.imread(car_image_path)
    if img is None:
        raise ValueError(f"Could not read image: {car_image_path}")
    
    # Predict license plates
    results = model.predict(source=car_image_path, conf=conf, verbose=False)
    
    cropped_plates = []
    
    for r in results:
        # Save annotated image if folder specified
        if ann_folder is not None:
            annotated_image = r.plot()
            os.makedirs(ann_folder, exist_ok=True)
            cv2.imwrite(f'{ann_folder}/{save_name}_Annotated.jpg', annotated_image)
        
        # Crop detected plates
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Extract crop
            crop = img[y1:y2, x1:x2]
            
            # Save crop if folder specified
            if crop_folder is not None:
                os.makedirs(crop_folder, exist_ok=True)
                crop_path = f'{crop_folder}/{save_name}_{i}.jpg'
                cv2.imwrite(crop_path, crop)
                cropped_plates.append(crop_path)
    
    return cropped_plates
