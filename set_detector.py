import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from ultralytics import YOLO
from itertools import combinations
from pathlib import Path
import os
import logging
import threading
import time
import gc

# Custom exception for model loading errors
class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass

# Configure logging with timestamps and file output
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler for persistent logs
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'set_detector.log'))
file_handler.setLevel(logging.INFO)

# Console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Format with timestamps and log levels
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configure TensorFlow for optimized performance
try:
    import multiprocessing
    CPU_COUNT = multiprocessing.cpu_count()
    THREAD_COUNT = min(2, CPU_COUNT)
    
    logger.info(f"Configuring TensorFlow with {THREAD_COUNT} threads (detected {CPU_COUNT} CPUs)")
    
    tf.config.threading.set_intra_op_parallelism_threads(THREAD_COUNT)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    tf.compat.v1.disable_eager_execution()
    
    # Enable memory growth for GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU detected and memory growth enabled for {len(gpus)} devices")
    else:
        logger.info("No GPU detected, using CPU optimizations")
        
except Exception as e:
    logger.warning(f"Failed to configure TensorFlow optimally: {e}")

# Cached models with TTL to prevent memory leaks
class ModelCache:
    def __init__(self, ttl=3600):  # 1-hour TTL by default
        self._cache = {}
        self._lock = threading.Lock()
        self._ttl = ttl
        self._last_cleanup = time.time()
        
    def get(self, key):
        with self._lock:
            self._cleanup()
            if key in self._cache:
                item = self._cache[key]
                item['timestamp'] = time.time()  # refresh timestamp
                return item['model']
            return None
            
    def set(self, key, model):
        with self._lock:
            self._cache[key] = {
                'model': model,
                'timestamp': time.time()
            }
            
    def _cleanup(self):
        now = time.time()
        if now - self._last_cleanup < 600:  # cleanup every 10 min
            return
            
        self._last_cleanup = now
        to_remove = []
        
        for key, item in self._cache.items():
            if now - item['timestamp'] > self._ttl:
                to_remove.append(key)
                
        for key in to_remove:
            del self._cache[key]
            
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} expired models from cache")
            gc.collect()

# Create model cache as module-level variable
_model_cache = ModelCache()

def debug_file_structure():
    """Helper function to debug file structure in Railway environment"""
    try:
        cwd = os.getcwd()
        logger.info(f"Current working directory: {cwd}")
        
        # List root directory
        root_contents = os.listdir(cwd)
        logger.info(f"Root directory contents: {root_contents}")
        
        # Check for models directory
        if 'models' in root_contents:
            models_path = os.path.join(cwd, 'models')
            models_contents = os.listdir(models_path)
            logger.info(f"Models directory contents: {models_contents}")
            
            # Check subdirectories
            for subdir in models_contents:
                subdir_path = os.path.join(models_path, subdir)
                if os.path.isdir(subdir_path):
                    logger.info(f"Contents of {subdir}: {os.listdir(subdir_path)}")
        
        # Check for model directories at root level
        for model_dir in ['Card', 'Characteristics', 'Shape']:
            if model_dir in root_contents:
                dir_path = os.path.join(cwd, model_dir)
                if os.path.isdir(dir_path):
                    logger.info(f"{model_dir} found at root level: {os.listdir(dir_path)}")
    
    except Exception as e:
        logger.error(f"Error in debug_file_structure: {e}")

def load_models():
    """Load models with caching and proper error handling."""
    # Check cache first
    models = (
        _model_cache.get('shape_model'), 
        _model_cache.get('fill_model'),
        _model_cache.get('detector_card'),
        _model_cache.get('detector_shape')
    )
    if all(models):
        logger.debug("Using cached models")
        return models
    
    # Print debug information about file structure
    debug_file_structure()
    
    logger.info("Loading models from disk")
    
    # First, check if models are in the app's root directory
    # For Railway deployment, models could be in multiple places:
    # 1. Direct repository root (same level as app.py)
    # 2. In a 'models' directory
    # 3. In a custom directory specified by MODELS_DIR
    
    # Try multiple potential locations for flexibility
    possible_locations = [
        Path(os.environ.get('MODELS_DIR', 'models')),  # Check env var first
        Path('models'),                                # Check models/ directory
        Path('.')                                      # Check app root directory
    ]
    
    base_dir = None
    for location in possible_locations:
        if (location / "Card").exists() and (location / "Characteristics").exists() and (location / "Shape").exists():
            base_dir = location
            logger.info(f"Found models at: {base_dir}")
            break
    
    if base_dir is None:
        # Log paths for debugging
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error(f"Directory contents: {os.listdir(os.getcwd())}")
        error_msg = "Model directories not found in any expected location"
        logger.error(error_msg)
        raise ModelLoadError(error_msg)
    
    char_path = base_dir / "Characteristics" / "11022025"
    shape_path = base_dir / "Shape" / "15052024" 
    card_path = base_dir / "Card" / "16042024"
    
    for path, name in [(char_path, 'Characteristics'), 
                       (shape_path, 'Shape'), 
                       (card_path, 'Card')]:
        if not path.exists():
            error_msg = f"Model directory {name} not found at {path}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
    
    try:
        shape_model_path = str(char_path / "shape_model.keras")
        fill_model_path = str(char_path / "fill_model.keras")
        detector_shape_path = str(shape_path / "best.pt")
        detector_card_path = str(card_path / "best.pt")
        
        missing_files = []
        for path in [shape_model_path, fill_model_path, detector_shape_path, detector_card_path]:
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            error_msg = f"Missing model files: {', '.join(missing_files)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
        
        # Load classification models
        start_time = time.time()
        logger.info("Loading shape classification model...")
        model_shape = load_model(shape_model_path)
        _model_cache.set('shape_model', model_shape)
        
        logger.info("Loading fill classification model...")
        model_fill = load_model(fill_model_path)
        _model_cache.set('fill_model', model_fill)
        
        # Load YOLO models, CPU only - Important for Railway
        logger.info("Loading shape detection model...")
        detector_shape = YOLO(detector_shape_path)
        detector_shape.conf = 0.5  # Match Colab example confidence threshold
        detector_shape.iou = 0.5   
        detector_shape.max_det = 15
        
        # Check for data.yaml file (used in Colab example)
        shape_yaml_path = str(shape_path / "data.yaml")
        if os.path.exists(shape_yaml_path):
            logger.info(f"Found shape data.yaml file, setting config")
            detector_shape.yaml = shape_yaml_path
        
        detector_shape.to("cpu")  # Ensure CPU usage for Railway compatibility
        _model_cache.set('detector_shape', detector_shape)
        
        logger.info("Loading card detection model...")
        detector_card = YOLO(detector_card_path)
        detector_card.conf = 0.5  # Match Colab example confidence threshold
        detector_card.iou = 0.5
        detector_card.max_det = 20
        
        # Check for data.yaml file (used in Colab example)
        card_yaml_path = str(card_path / "data.yaml")
        if os.path.exists(card_yaml_path):
            logger.info(f"Found card data.yaml file, setting config")
            detector_card.yaml = card_yaml_path
            
        detector_card.to("cpu")  # Ensure CPU usage for Railway compatibility
        _model_cache.set('detector_card', detector_card)
        
        logger.info(f"All models loaded successfully in {time.time() - start_time:.2f} seconds")
        return model_shape, model_fill, detector_card, detector_shape
    
    except ModelLoadError:
        raise
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}", exc_info=True)
        raise ModelLoadError(f"Failed to load models: {str(e)}")

# The rest of the code remains the same as it focuses on core functionality
# I'll include key functions but the rest is unchanged

def correct_orientation(board_image, card_detector):
    """Rotate image if cards are vertical, with optimized processing."""
    h, w = board_image.shape[:2]
    max_dim = 1200
    
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        board_image_small = cv2.resize(board_image, (new_w, new_h))
        detection = card_detector(board_image_small)
    else:
        detection = card_detector(board_image)
    
    boxes = detection[0].boxes.xyxy.cpu().numpy().astype(int)
    if boxes.size == 0: 
        logger.warning("No cards detected during orientation check")
        return board_image, False
    
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    
    width_mean = np.mean(widths)
    height_mean = np.mean(heights)
    
    is_vertical = height_mean > width_mean
    if is_vertical:
        logger.info("Rotating image 90° clockwise (detected vertical layout)")
        return cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE), True
    else:
        return board_image, False

def restore_orientation(img, was_rotated):
    """Restore original orientation if needed."""
    if was_rotated:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def predict_color(img_bgr):
    """Classify color using HSV thresholds with enhanced accuracy."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    mask_green = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([85, 255, 255]))
    mask_purple = cv2.inRange(hsv, np.array([115, 40, 40]), np.array([165, 255, 255]))
    
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    if total_pixels == 0:
        return "unknown"
        
    counts = {
        "green": cv2.countNonZero(mask_green) / total_pixels,
        "purple": cv2.countNonZero(mask_purple) / total_pixels, 
        "red": cv2.countNonZero(mask_red) / total_pixels
    }
    
    max_color = max(counts, key=counts.get)
    if counts[max_color] < 0.05:
        return "unknown"
        
    return max_color

# Note: The rest of the functions remain the same as in the original code
# Including detect_cards, predict_card_features, classify_cards_on_board, 
# valid_set, locate_all_sets, draw_set_indicators, identify_sets

# For brevity, I'm only showing the key function that would need adjustment

def identify_sets(image):
    """
    Complete pipeline to find SETs in an image.
    This is the main entry point for the SET detection functionality.
    """
    start_time = time.time()
    logger.info(f"Starting SET detection on image of shape {image.shape}")
    
    try:
        model_shape, model_fill, detector_card, detector_shape = load_models()
        
        h, w = image.shape[:2]
        max_dim = 1500
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            logger.info(f"Resized input image from {(w, h)} to {(new_w, new_h)} for CPU efficiency")
        
        try:
            image_enhanced = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        except Exception as e:
            logger.warning(f"Error during image enhancement, using original image: {e}")
            image_enhanced = image
        
        try:
            processed, was_rotated = correct_orientation(image_enhanced, detector_card)
        except Exception as e:
            logger.warning(f"Error during orientation correction: {e}. Using original orientation.")
            processed, was_rotated = image_enhanced, False
        
        # The rest of the processing pipeline follows
        # For Railway deployment, we ensure efficient memory usage
        
        df_cards = classify_cards_on_board(processed, detector_card, detector_shape, model_fill, model_shape)
        if df_cards.empty:
            logger.warning("No valid cards detected in the image")
            return [], image
        
        found_sets = locate_all_sets(df_cards)
        if found_sets:
            annotated = draw_set_indicators(processed.copy(), found_sets)
            final_image = restore_orientation(annotated, was_rotated)
            logger.info(f"SET detection complete. Found {len(found_sets)} SETs in {time.time() - start_time:.2f} seconds")
            return found_sets, final_image
        else:
            logger.info(f"No SETs found in the image after {time.time() - start_time:.2f} seconds")
            return [], restore_orientation(processed, was_rotated)
            
    except ModelLoadError as e:
        logger.error(f"Model loading error in SET detection: {str(e)}")
        return [], image
    except Exception as e:
        logger.error(f"Unexpected error in SET detection: {str(e)}", exc_info=True)
        return [], image
