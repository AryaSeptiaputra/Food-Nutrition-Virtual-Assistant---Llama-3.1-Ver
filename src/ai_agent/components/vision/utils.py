import hashlib
from pathlib import Path
import datetime
import uuid
import json
import shutil
import cv2
import os

def generate_base_filename(prefix: str = "upload") -> str:
    """Generate a unique base filename with timestamp and UUID.
    
    Args:
        prefix: String prefix for the filename (default: "upload")
    
    Returns:
        A unique filename in format: {prefix}_{YYYYMMDD_HHMMSS}_{4-char-UUID}
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:4]
    return f"{prefix}_{timestamp}_{unique}"

def hash_file(file_path: str):
    """Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to the file to hash
    
    Returns:
        Hexadecimal hash string if file exists, None otherwise
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def is_image_changed(current_path: str, last_hash_path: str) -> bool:
    """Check if an image file has changed by comparing its hash.
    
    Compares the current file's hash with a previously stored hash.
    Updates the stored hash file if the image has changed.
    
    Args:
        current_path: Path to the current image file
        last_hash_path: Path to the file storing the last known hash
    
    Returns:
        True if image has changed, False if unchanged or file doesn't exist
    """
    current_hash = hash_file(current_path)
    if current_hash is None: return False

    last_hash_file = Path(last_hash_path)
    last_hash_file.parent.mkdir(parents=True, exist_ok=True)

    if last_hash_file.exists():
        previous_hash = last_hash_file.read_text().strip()
        if previous_hash == current_hash:
            return False

    last_hash_file.write_text(current_hash)
    return True

def validate_image_format(file_path: str) -> str | None:
    """Validate if a file is a valid image format and can be read.
    
    Performs three checks: file existence, file extension, and image integrity.
    Checks if the file exists, has a valid image extension, and can be read by OpenCV.
    
    Args:
        file_path: Path to the file to validate
    
    Returns:
        Full path string if file is valid, None if file doesn't exist, has invalid extension, or cannot be read as image
    """

    if not file_path:
        return None

    path = Path(file_path)
    
    if not path.exists(): 
        return None
        
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    if path.suffix.lower() not in valid_extensions: 
        return None
        
    try:
        img = cv2.imread(str(path))
        if img is None: 
            return None
    except Exception: 
        return None
        
    return str(path)

def remove_file(file_path: str):
    """Remove a file from the system with error handling.
    
    Safely deletes a file from the filesystem. If the file doesn't exist,
    it is silently ignored. Any deletion errors are printed but not raised.
    
    Args:
        file_path: Path to the file to remove
    """
    try:
        path = Path(file_path)
        if path.exists():
            os.remove(path)
    except Exception as e:
        print(f"Error removing file {file_path}: {e}")

def store_raw_image(image_path: str, images_dir: Path, base_name: str) -> Path:
    """Copy a raw image file to the storage directory.
    
    Args:
        image_path: Path to the source image file
        images_dir: Directory path where the image will be stored
        base_name: Base name for the stored image (extension will be appended)
    
    Returns:
        Path to the stored image file
    """
    original = Path(image_path)
    ext = original.suffix
    raw_path = images_dir / f"{base_name}{ext}"
    shutil.copy(original, raw_path)
    return raw_path

def store_annotated_image(result_obj, annotated_dir: Path, base_name: str) -> Path:
    """Store an annotated image with detection overlays.
    
    Takes a YOLO result object and saves the plotted annotation as a JPEG image.
    
    Args:
        result_obj: YOLO detection result object with plot() method
        annotated_dir: Directory path where the annotated image will be stored
        base_name: Base name for the output image (will append "_annotated.jpg")
    
    Returns:
        Path to the stored annotated image file
    """
    annotated_path = annotated_dir / f"{base_name}_annotated.jpg"
    annotated_img = result_obj.plot()
    cv2.imwrite(str(annotated_path), annotated_img)
    return annotated_path

def load_cache(cache_path: Path) -> dict:
    """Load cache data from a JSON file.
    
    Args:
        cache_path: Path to the cache JSON file
    
    Returns:
        Dictionary containing cache data, or empty dict if file doesn't exist or is invalid
    """
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except: return {}
    return {}

def save_cache(cache_path: Path, cache: dict):
    """Save cache data to a JSON file with indentation.
    
    Args:
        cache_path: Path where the cache JSON file will be saved
        cache: Dictionary containing the cache data to persist
    """
    cache_path.write_text(json.dumps(cache, indent=2))