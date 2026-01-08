"""Main entry point for Car Detection Pipeline"""

import os
import glob
from pathlib import Path
import typer
from typing import Optional
from src.pipeline import CarDetectionPipeline

app = typer.Typer(help="Car Detection Pipeline: Detect cars, seatbelts, and license plates from images")


def get_image_paths(input_path: str) -> list:
    """
    Get list of image paths from input path.
    
    Args:
        input_path: Path to a single image file or directory containing images
    
    Returns:
        List of image file paths
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single image file
        return [str(input_path)]
    elif input_path.is_dir():
        # Directory with images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(str(input_path / ext)))
            image_paths.extend(glob.glob(str(input_path / ext.upper())))
        return sorted(image_paths)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


@app.command()
def main(
    input_path: str = typer.Argument(..., help="Path to input image file or directory containing images"),
    output: str = typer.Option("output", "--output", "-o", help="Directory path for output JSON file and intermediate results (default: 'output')"),
    json_name: str = typer.Option("results.json", "--json-name", "-j", help="Name of the output JSON file (default: 'results.json')"),
    violators_json_name: str = typer.Option("violators.json", "--violators-json-name", "-v", help="Name of the violators JSON file (default: 'violators.json')"),
    car_conf: float = typer.Option(0.4, "--car-conf", help="Confidence threshold for car detection (default: 0.4)"),
    seatbelt_conf: float = typer.Option(0.55, "--seatbelt-conf", help="Confidence threshold for seatbelt detection (default: 0.55)"),
    plate_conf: float = typer.Option(0.4, "--plate-conf", help="Confidence threshold for license plate detection (default: 0.4)"),
    car_model: str = typer.Option("yolo11n.pt", "--car-model", help="Path to car detection YOLO model (default: 'yolo11n.pt')"),
    seatbelt_model: str = typer.Option("seat_belt_fast/10_epoch_m3/weights/best.pt", "--seatbelt-model", help="Path to seatbelt detection YOLO model"),
    plate_model: str = typer.Option("license-plate-finetune-v1n.pt", "--plate-model", help="Path to license plate detection YOLO model"),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Use GPU for EasyOCR (default: True)")
):
    """
    Car Detection Pipeline: Detect cars, seatbelts, and license plates from images.
    
    This pipeline processes images to:
    1. Detect and crop cars
    2. Detect seatbelts in each car
    3. Detect and extract license plate text from each car
    
    Results are saved as JSON with image names as keys.
    """
    # Validate input path
    if not os.path.exists(input_path):
        typer.echo(f"Error: Input path does not exist: {input_path}", err=True)
        raise typer.Exit(1)
    
    # Get image paths
    typer.echo(f"Getting images from: {input_path}")
    image_paths = get_image_paths(input_path)
    
    if not image_paths:
        typer.echo(f"Error: No images found in: {input_path}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Found {len(image_paths)} image(s) to process")
    
    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)
    
    # Initialize pipeline
    typer.echo("\n" + "="*60)
    typer.echo("Initializing Car Detection Pipeline")
    typer.echo("="*60)
    pipeline = CarDetectionPipeline(
        car_model_path=car_model,
        seatbelt_model_path=seatbelt_model,
        plate_model_path=plate_model,
        car_conf=car_conf,
        seatbelt_conf=seatbelt_conf,
        plate_conf=plate_conf,
        output_dir=output,
        gpu=gpu
    )
    
    # Process all images
    typer.echo("\n" + "="*60)
    typer.echo("Processing Images")
    typer.echo("="*60)
    results = pipeline.process_images(image_paths)
    
    # Generate JSON output
    json_output_path = os.path.join(output, json_name)
    typer.echo("\n" + "="*60)
    typer.echo("Generating JSON Output")
    typer.echo("="*60)
    pipeline.generate_json(results, json_output_path)
    
    # Generate violators JSON output
    violators_json_output_path = os.path.join(output, violators_json_name)
    typer.echo("\n" + "="*60)
    typer.echo("Generating Violators JSON Output")
    typer.echo("="*60)
    pipeline.generate_violators_json(results, violators_json_output_path)
    
    typer.echo("\n" + "="*60)
    typer.echo("Pipeline Completed Successfully!")
    typer.echo("="*60)
    typer.echo(f"Results saved to: {json_output_path}")
    typer.echo(f"Violators saved to: {violators_json_output_path}")
    typer.echo(f"Intermediate outputs saved to: {output}")


if __name__ == "__main__":
    app()
