"""Text Extraction Module for License Plates"""

from typing import Dict, Optional


def extract_plate_text(
    plate_image_path: str,
    reader,
    gpu: bool = True
) -> Dict[str, str]:
    """
    Extract text from a license plate image using EasyOCR.
    
    Args:
        plate_image_path: Path to the license plate image
        reader: EasyOCR Reader instance (should be initialized once and reused)
        gpu: Whether to use GPU for OCR (default: True, only used if reader is None)
    
    Returns:
        Dictionary with 'text' and 'avg_confidence' keys.
        If no text detected: {'text': 'None', 'avg_confidence': 'None'}
        If text detected: {'text': 'extracted_text', 'avg_confidence': '0.XX'}
    """
    # Read text from image
    results = reader.readtext(plate_image_path)
    
    # If no text detected
    if len(results) == 0:
        return {
            'text': 'None',
            'avg_confidence': 'None'
        }
    
    # Extract texts and calculate average confidence
    texts = []
    confidence_sum = 0.0
    
    for (bbox, text, confidence) in results:
        texts.append(text)
        confidence_sum += confidence
    
    avg_confidence = confidence_sum / len(results)
    
    return {
        'text': ','.join(texts),
        'avg_confidence': str(avg_confidence)
    }
