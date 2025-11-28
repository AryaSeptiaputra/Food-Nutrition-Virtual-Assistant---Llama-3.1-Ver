import os
import shutil
import cv2
import numpy as np
import time
from pathlib import Path

from src.ai_agent.components.vision.inferance import VisionInference 
from src.config.settings import settings

def create_dummy_image(filename, color, size=(640, 640)):
    """Create a solid-colored dummy image for testing purposes.
    
    Args:
        filename: Output filename for the test image
        color: Tuple of BGR color values (e.g., (0, 0, 255) for red)
        size: Tuple of (width, height) for the image (default: 640x640)
    
    Returns:
        The filename of the created image
    """
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    image[:] = color
    
    cv2.putText(image, f"Test {filename}", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imwrite(filename, image)
    print(f"[SETUP] Dummy image created: {filename}")
    return filename

def simulate_upload_and_process(vision_system, source_image_path):
    """Simulate user uploading and processing an image through the vision system.
    
    Copies the source image to temp directory before processing, as the vision system
    automatically deletes the input file after completion. Measures processing time
    and displays cache hit/miss status.
    
    Args:
        vision_system: VisionInference system instance
        source_image_path: Path to the source image file to process
    """
    if not os.path.exists(source_image_path):
        print(f"[ERROR] Source file not found: {source_image_path}")
        return

    temp_dir = Path(settings.CACHE_PATH)
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True)

    simulated_upload_path = temp_dir / f"upload_{Path(source_image_path).name}"
    
    shutil.copy(source_image_path, simulated_upload_path)
    
    print(f"\nProcessing: {source_image_path}")
    print(f"   (Temp Input: {simulated_upload_path})")

    start_time = time.time()
    result = vision_system.detected(str(simulated_upload_path))
    end_time = time.time()

    print(f"   [Processing Time]: {end_time - start_time:.4f} seconds")
    
    if result.get('raw') is None:
        print(f"   [Status]: {result.get('result')}")
    else:
        changed_status = result.get('changed')
        status_text = "NEW (YOLO Inference)" if changed_status else "CACHED (From Cache)"
        print(f"   [Mode]: {status_text}")
        print(f"   [Hash Changed]: {changed_status}")
        print(f"   [Result]: {result.get('result')}")
        print(f"   [File Raw]: {result.get('raw')}")

    if not simulated_upload_path.exists():
        print("   [Cleanup]: Success (Temp input file automatically deleted)")
    else:
        print("   [Cleanup]: FAILED (Temp file still exists!)")

def main():
    """Test vision inference system with multiple test scenarios.
    
    Runs five test cases:
    1. Red image (first time) - should run inference
    2. Red image (repeated) - should hit cache
    3. Blue image (new image) - should run inference
    4. Red image again - should hit cache
    5. Text file (invalid format) - should fail validation
    
    Demonstrates cache behavior, file cleanup, and error handling.
    """
    print("Initializing Vision System")
    vision = VisionInference(
        model_path=settings.YOLO_MODEL_PATH,
        images_dir=settings.IMAGES_DIR,
        cache_dir=settings.CACHE_PATH,
        annotated_dir=settings.ANNOTATED_DIR
    )

    image_red = create_dummy_image("test_red.jpg", (0, 0, 255))
    image_blue = create_dummy_image("test_blue.jpg", (255, 0, 0))
    
    test_file = "test_fake.txt"
    with open(test_file, "w") as f: 
        f.write("This is not an image")

    try:
        print("\nTEST 1: Red Image (First Time)")
        simulate_upload_and_process(vision, image_red)

        print("\nTEST 2: Red Image (Repeated Immediately)")
        simulate_upload_and_process(vision, image_red)

        print("\nTEST 3: Blue Image (New Image)")
        simulate_upload_and_process(vision, image_blue)

        print("\nTEST 4: Red Image Again (After Blue)")
        simulate_upload_and_process(vision, image_red)

        print("\nTEST 5: Upload Text File (.txt)")
        simulate_upload_and_process(vision, test_file)

    finally:
        print("\nCleaning Up Local Dummy Files")
        for test_image in [image_red, image_blue, test_file]:
            if os.path.exists(test_image):
                os.remove(test_image)
        print("Done.")

if __name__ == "__main__":
    main()