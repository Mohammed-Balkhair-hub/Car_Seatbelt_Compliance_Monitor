"""Car Detection Module"""

import cv2
import os
from pathlib import Path
from typing import List, Tuple, Optional


def detect_and_crop_cars(
    image_path: str,
    model,
    conf: float = 0.4,
    ann_folder: Optional[str] = None,
    crop_folder: Optional[str] = None,
    save_name: Optional[str] = None
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Detect and crop cars from an image using YOLO model.
    
    Args:
        image_path: Path to the input image
        model: YOLO model instance
        conf: Confidence threshold for car detection (default: 0.4)
        ann_folder: Directory to save annotated images (optional)
        crop_folder: Directory to save cropped car images (optional)
        save_name: Base name for saved files (optional, will use image filename if not provided)
    
    Returns:
        List of tuples: [(cropped_image_path, (x1, y1, x2, y2)), ...]
        Empty list if no cars detected.
    """
    # Get image name for saving if not provided
    if save_name is None:
        save_name = Path(image_path).stem
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Predict cars (class 2)
    results = model.predict(source=image_path, classes=[2], conf=conf, imgsz=640, verbose=False)
    
    cropped_cars = []
    
    for r in results:
        # Save annotated image if folder specified
        if ann_folder is not None:
            annotated_image = r.plot()
            os.makedirs(ann_folder, exist_ok=True)
            cv2.imwrite(f'{ann_folder}/{save_name}_Annotated.jpg', annotated_image)
        
        # Crop detected cars
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
                cropped_cars.append((crop_path, (x1, y1, x2, y2)))
            else:
                # Return in-memory crop info if no save folder
                cropped_cars.append((None, (x1, y1, x2, y2)))
    
    return cropped_cars
