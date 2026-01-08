"""Pipeline Orchestration Module"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO
import easyocr

from ..car_detection.detector import detect_and_crop_cars
from ..seatbelt_detection.detector import detect_seatbelt
from ..license_plate.detector import detect_and_crop_plates
from ..license_plate.text_extractor import extract_plate_text


class CarDetectionPipeline:
    """Main pipeline for car detection, seatbelt detection, and license plate extraction."""
    
    def __init__(
        self,
        car_model_path: str = "yolo11n.pt",
        seatbelt_model_path: str = "seat_belt_fast/10_epoch_m3/weights/best.pt",
        plate_model_path: str = "license-plate-finetune-v1n.pt",
        car_conf: float = 0.4,
        seatbelt_conf: float = 0.55,
        plate_conf: float = 0.4,
        output_dir: str = "output",
        gpu: bool = True
    ):
        """
        Initialize the car detection pipeline.
        
        Args:
            car_model_path: Path to car detection YOLO model
            seatbelt_model_path: Path to seatbelt detection YOLO model
            plate_model_path: Path to license plate detection YOLO model
            car_conf: Confidence threshold for car detection
            seatbelt_conf: Confidence threshold for seatbelt detection
            plate_conf: Confidence threshold for license plate detection
            output_dir: Base directory for all outputs
            gpu: Whether to use GPU for EasyOCR
        """
        # Load models
        print(f"Loading car detection model: {car_model_path}")
        self.car_model = YOLO(car_model_path)
        
        print(f"Loading seatbelt detection model: {seatbelt_model_path}")
        self.seatbelt_model = YOLO(seatbelt_model_path)
        
        print(f"Loading license plate detection model: {plate_model_path}")
        self.plate_model = YOLO(plate_model_path)
        
        # Initialize EasyOCR reader (once, reused)
        print("Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        print("EasyOCR reader initialized.")
        
        # Set confidence thresholds
        self.car_conf = car_conf
        self.seatbelt_conf = seatbelt_conf
        self.plate_conf = plate_conf
        
        # Set output directories
        self.output_dir = output_dir
        self.cars_ann_dir = os.path.join(output_dir, "cars_annotated")
        self.cars_crop_dir = os.path.join(output_dir, "cars_cropped")
        self.seatbelt_ann_dir = os.path.join(output_dir, "seatbelt_annotated")
        self.plates_ann_dir = os.path.join(output_dir, "plates_annotated")
        self.plates_crop_dir = os.path.join(output_dir, "plates_cropped")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_image(self, image_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a single image through the entire pipeline.
        
        Args:
            image_path: Path to the input image
        
        Returns:
            Dictionary with image name as key and list of car results as value.
            Format: {image_name: [{"plate": "...", "seatbelt": bool}, ...]}
        """
        image_name = Path(image_path).name
        image_stem = Path(image_path).stem
        
        print(f"\nProcessing image: {image_name}")
        
        # Step 1: Detect and crop cars
        print(f"  Detecting cars (conf={self.car_conf})...")
        cropped_cars = detect_and_crop_cars(
            image_path=image_path,
            model=self.car_model,
            conf=self.car_conf,
            ann_folder=self.cars_ann_dir,
            crop_folder=self.cars_crop_dir,
            save_name=image_stem
        )
        
        if not cropped_cars:
            print(f"  No cars detected in {image_name}")
            return {image_name: []}
        
        print(f"  Found {len(cropped_cars)} car(s)")
        
        # Step 2: Process each cropped car
        car_results = []
        for car_idx, (car_crop_path, bbox) in enumerate(cropped_cars):
            if car_crop_path is None:
                continue
            
            car_stem = Path(car_crop_path).stem
            print(f"    Processing car {car_idx + 1}/{len(cropped_cars)}: {car_stem}")
            
            # Step 2a: Detect seatbelt
            print(f"      Detecting seatbelt (conf={self.seatbelt_conf})...")
            has_seatbelt = detect_seatbelt(
                car_image_path=car_crop_path,
                model=self.seatbelt_model,
                conf=self.seatbelt_conf,
                ann_folder=self.seatbelt_ann_dir,
                save_name=car_stem
            )
            print(f"      Seatbelt detected: {has_seatbelt}")
            
            # Step 2b: Detect and crop license plates
            print(f"      Detecting license plates (conf={self.plate_conf})...")
            cropped_plates = detect_and_crop_plates(
                car_image_path=car_crop_path,
                model=self.plate_model,
                conf=self.plate_conf,
                ann_folder=self.plates_ann_dir,
                crop_folder=self.plates_crop_dir,
                save_name=car_stem
            )
            
            if not cropped_plates:
                print(f"      No license plates detected")
                # Still add car result with no plate
                car_results.append({
                    "plate": "None",
                    "seatbelt": has_seatbelt
                })
            else:
                print(f"      Found {len(cropped_plates)} plate(s)")
                
                # Step 2c: Extract text from all detected plates
                plate_texts = []
                for plate_path in cropped_plates:
                    print(f"        Extracting text from {Path(plate_path).name}...")
                    text_result = extract_plate_text(
                        plate_image_path=plate_path,
                        reader=self.reader
                    )
                    plate_text = text_result['text']
                    print(f"        Extracted text: {plate_text}")
                    
                    # Use first non-None plate text, or combine all
                    if plate_text != 'None':
                        plate_texts.append(plate_text)
                
                # Use first detected plate text, or 'None' if no text extracted
                final_plate_text = plate_texts[0] if plate_texts else "None"
                
                car_results.append({
                    "plate": final_plate_text,
                    "seatbelt": has_seatbelt
                })
        
        return {image_name: car_results}
    
    def process_images(self, image_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process multiple images through the pipeline.
        
        Args:
            image_paths: List of paths to input images
        
        Returns:
            Dictionary with image names as keys and list of car results as values.
            Format: {image_name: [{"plate": "...", "seatbelt": bool}, ...]}
        """
        all_results = {}
        
        for image_path in image_paths:
            result = self.process_image(image_path)
            all_results.update(result)
        
        return all_results
    
    def filter_violators(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Filter results to only include violators (cars without seatbelts that have plate text).
        
        A violator is defined as a car where:
        - seatbelt: false
        - AND plate != "None" (has actual plate text)
        
        Args:
            results: Dictionary with full results from process_image or process_images
        
        Returns:
            Dictionary containing only violators, with same structure as input.
            Images with no violators are excluded.
        """
        violators = {}
        
        for image_name, cars in results.items():
            # Filter cars to only include violators
            violator_cars = [
                car for car in cars
                if car.get("seatbelt") is False and car.get("plate") != "None"
            ]
            
            # Only add image if it has at least one violator
            if violator_cars:
                violators[image_name] = violator_cars
        
        return violators
    
    def generate_json(self, results: Dict[str, List[Dict[str, Any]]], output_path: str):
        """
        Generate JSON file from pipeline results.
        
        Args:
            results: Dictionary with results from process_image or process_images
            output_path: Path where JSON file should be saved
        """
        with open(output_path, 'w') as fp:
            json.dump(results, fp, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    def generate_violators_json(
        self, 
        results: Dict[str, List[Dict[str, Any]]], 
        output_path: str
    ):
        """
        Generate JSON file containing only violators (cars without seatbelts with plate text).
        
        Args:
            results: Dictionary with full results from process_image or process_images
            output_path: Path where violators JSON file should be saved
        """
        violators = self.filter_violators(results)
        
        with open(output_path, 'w') as fp:
            json.dump(violators, fp, indent=2)
        
        violator_count = sum(len(cars) for cars in violators.values())
        image_count = len(violators)
        print(f"\nViolators JSON saved to: {output_path}")
        print(f"Found {violator_count} violator(s) across {image_count} image(s)")