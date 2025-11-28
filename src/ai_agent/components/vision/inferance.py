import cv2
from pathlib import Path
from ultralytics import YOLO

from src.ai_agent.components.vision.utils import (
    generate_base_filename,
    is_image_changed,
    hash_file,
    store_raw_image,
    store_annotated_image,
    load_cache,
    save_cache,
    validate_image_format,
    remove_file
)

class VisionInference:

    def __init__(self, model_path: str, raw_dir: str, cache_dir: str, annotated_dir: str, temp_dir: str):
        """Initialize the Vision Inference system with YOLO model and directory paths.
        
        Sets up directory structure for storing raw images, annotated images, cache, and temporary files.
        Loads a YOLO model from disk and initializes cache and hash tracking files.
        
        Args:
            model_path: Path to the YOLO model weights file (.pt)
            raw_dir: Directory to store original uploaded images
            cache_dir: Directory to store detection cache and hash tracking
            annotated_dir: Directory to store YOLO-annotated images with bounding boxes
            temp_dir: Directory for temporary files during processing
        """
        self.model_path = model_path
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir)
        self.annotated_dir = Path(annotated_dir)
        self.temp_dir = Path(temp_dir)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.last_hash_path = self.cache_dir / ".hash"
        self.cache_path = self.cache_dir / "cache.json"

        self.model = YOLO(self.model_path)


    def inference(self, image_path: str) -> list:
        """Run YOLO object detection on an image and extract detected class names.
        
        Performs food detection inference using the pre-loaded YOLO model.
        Extracts unique class names from detection results.
        
        Args:
            image_path: Path to the image file to perform inference on
        
        Returns:
            List of unique detected food class names. Empty list if no objects detected.
        """
        image = cv2.imread(str(image_path))

        results = self.model.predict(image, conf=0.25, iou=0.45, verbose=False, max_det=20)
        result = results[0]
        boxes = result.boxes

        if boxes is None: 
            return []
        
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = list({result.names[c] for c in class_ids})
        return class_names

    def detected(self, image_path: str = None) -> dict:
        """Detect foods in an image with caching and duplicate detection prevention.
        
        Main entry point for food detection. Validates image format, checks cache for duplicates,
        runs YOLO inference if needed, and stores results with annotated images.
        Automatically removes input file after processing (cleanup in finally block).
        
        Args:
            image_path: Path to image file (optional; scans temp_dir if not provided)
        
        Returns:
            Dictionary with keys:
            - 'changed': Boolean indicating if new detection was performed (vs cache hit)
            - 'result': String description of detected foods or error message
            - 'raw': Path to stored original image, or None if no image provided
            - 'annotated': Path to stored annotated image, or None if no image provided
        
        Processing flow:
        1. Validates image format (checks extensions and readability)
        2. Scans temp_dir if no image provided
        3. Computes file hash and checks cache for duplicates
        4. Returns cached result if image unchanged
        5. Runs YOLO inference for new/changed images
        6. Stores raw and annotated images
        7. Updates cache with new results
        8. Cleans up input file in finally block
        """
        valid_path = validate_image_format(image_path)

        if not valid_path:
             for file in self.temp_dir.iterdir():
                 validated_file = validate_image_format(str(file))
                 if validated_file:
                     valid_path = validated_file
                     break
        
        if not valid_path:
             return {"result": "the user did not provide an image.", "raw": None, "annotated": None}

        image_path_str = valid_path
        current_cache = load_cache(self.cache_path)

        try:
            file_hash = hash_file(image_path_str)
            if file_hash is None:
                return {"result": "the user did not provide an image.", "raw": None, "annotated": None}

            image_modified = is_image_changed(image_path_str, str(self.last_hash_path))
            
            if not image_modified and file_hash in current_cache:
                return {
                    "changed": False,
                    **current_cache[file_hash]
                }

            base_name = generate_base_filename()
            results = self.model.predict(image_path_str, verbose=False)
            result_obj = results[0]
            detected_classes = self.inference(image_path_str)

            raw_path = store_raw_image(image_path_str, self.raw_dir, base_name)
            annotated_path = store_annotated_image(result_obj, self.annotated_dir, base_name)

            current_cache[file_hash] = {
                "changed": True,
                "result": f"{detected_classes} are visible in the image.",
                "raw": str(raw_path),
                "annotated": str(annotated_path)
            }

            save_cache(self.cache_path, current_cache)

            return current_cache[file_hash]

        except Exception as e:
            print(f"Error processing detection: {e}")
            return {"result": "Error occurred during YOLO processing.", "raw": None, "annotated": None}

        finally:
            remove_file(image_path_str)
