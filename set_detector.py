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
import sys

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
    
    # Railway has limited CPU resources, so we need to be careful not to overuse them
    tf.config.threading.set_intra_op_parallelism_threads(THREAD_COUNT)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Disable eager execution for better performance
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
    """
    Load ML models with improved path resolution for Railway deployment.
    
    Enhanced with multiple fallback paths and better error handling.
    """
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
    
    logger.info("Loading models from disk")
    
    # Debug the environment
    debug_file_structure()
    
    # Get current working directory
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    
    # Try to read MODELS_DIR from environment with a fallback
    models_dir_env = os.environ.get('MODELS_DIR', '/app/models')
    logger.info(f"MODELS_DIR environment variable: {models_dir_env}")
    
    # Define base paths with Railway-friendly path resolution
    app_root = Path(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"App root directory: {app_root}")
    
    # Try multiple potential locations for flexibility
    possible_locations = [
        Path(models_dir_env),             # First check environment variable
        Path(cwd) / "models",             # Check models in current working directory (Railway)
        app_root / "models",              # Check ./models/ directory
        app_root.parent / "models",       # Check parent directory
        Path("/app/models"),              # Docker default location
        Path(cwd)                         # Last resort: current directory
    ]
    
    # Debug: Log all potential paths we're checking
    for idx, loc in enumerate(possible_locations):
        logger.info(f"Checking location {idx+1} for models: {loc}")
        try:
            if loc.exists():
                logger.info(f"Directory exists. Contents: {list(loc.glob('*'))}")
                # Check deeper for subdirectories
                for subdir in ['Card', 'Characteristics', 'Shape']:
                    subpath = loc / subdir
                    if subpath.exists():
                        logger.info(f"Found {subdir} directory at {subpath}")
                        logger.info(f"Contents: {list(subpath.glob('*'))}")
            else:
                logger.info(f"Directory does not exist: {loc}")
        except Exception as e:
            logger.warning(f"Error checking path {loc}: {str(e)}")
    
    # Find the first valid path that contains the expected model directories
    base_dir = None
    for location in possible_locations:
        if not location.exists():
            continue
            
        # Check if this location has the expected model subdirectories
        card_dir = location / "Card"
        char_dir = location / "Characteristics"
        shape_dir = location / "Shape"
        
        if card_dir.exists() and char_dir.exists() and shape_dir.exists():
            base_dir = location
            logger.info(f"Found models at: {base_dir}")
            break
            
        # Check if we have model directories with version numbers
        card_versions = [d for d in location.glob("Card/*") if d.is_dir()]
        char_versions = [d for d in location.glob("Characteristics/*") if d.is_dir()]
        shape_versions = [d for d in location.glob("Shape/*") if d.is_dir()]
        
        if card_versions and char_versions and shape_versions:
            base_dir = location
            logger.info(f"Found models with versioned directories at: {base_dir}")
            break
    
    if base_dir is None:
        # Last fallback - check if we have nested models directory
        for nested_path in [
            Path(cwd) / "models" / "models",
            Path("/app") / "models" / "models"
        ]:
            if nested_path.exists() and any([
                (nested_path / subdir).exists() for subdir in ['Card', 'Characteristics', 'Shape']
            ]):
                base_dir = nested_path
                logger.info(f"Found models in nested directory: {base_dir}")
                break
                
        if base_dir is None:
            # Log paths for debugging
            error_msg = "Model directories not found in any expected location"
            logger.error(error_msg)
            try:
                # Try to show file tree for debugging
                import subprocess
                result = subprocess.run(["find", str(cwd), "-type", "f", "-name", "*.pt", "-o", "-name", "*.keras"], 
                                      capture_output=True, text=True)
                logger.error(f"File search results: {result.stdout}")
            except Exception as e:
                logger.error(f"Error during file search: {e}")
            raise ModelLoadError(error_msg)
    
    # Find specific model paths with version detection
    # First check for versioned directories
    card_versions = list(base_dir.glob("Card/*/"))
    char_versions = list(base_dir.glob("Characteristics/*/"))
    shape_versions = list(base_dir.glob("Shape/*/"))
    
    # If versions found, use the latest one (or specified one)
    card_path = (base_dir / "Card" / "16042024" if (base_dir / "Card" / "16042024").exists() 
                else (card_versions[-1] if card_versions else None))
                
    char_path = (base_dir / "Characteristics" / "11022025" if (base_dir / "Characteristics" / "11022025").exists() 
                else (char_versions[-1] if char_versions else None))
                
    shape_path = (base_dir / "Shape" / "15052024" if (base_dir / "Shape" / "15052024").exists() 
                else (shape_versions[-1] if shape_versions else None))
    
    if not all([card_path, char_path, shape_path]):
        error_msg = f"Missing model directories. Found: card={card_path}, char={char_path}, shape={shape_path}"
        logger.error(error_msg)
        raise ModelLoadError(error_msg)
    
    logger.info(f"Using model paths: Card={card_path}, Characteristics={char_path}, Shape={shape_path}")
    
    try:
        # Define file paths for each model
        shape_model_path = str(char_path / "shape_model.keras")
        fill_model_path = str(char_path / "fill_model.keras")
        detector_shape_path = str(shape_path / "best.pt")
        detector_card_path = str(card_path / "best.pt")
        
        # Check that each file exists
        missing_files = []
        for path, name in [
            (shape_model_path, "shape_model.keras"),
            (fill_model_path, "fill_model.keras"),
            (detector_shape_path, "best.pt (Shape)"),
            (detector_card_path, "best.pt (Card)")
        ]:
            if not os.path.exists(path):
                missing_files.append(f"{name} at {path}")
        
        if missing_files:
            error_msg = f"Missing model files: {', '.join(missing_files)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
        
        # Load classification models
        start_time = time.time()
        logger.info("Loading shape classification model...")
        model_shape = load_model(shape_model_path)
        _model_cache.set('shape_model', model_shape)
        logger.info(f"Shape model loaded successfully in {time.time() - start_time:.2f} seconds")
        
        logger.info("Loading fill classification model...")
        start_time = time.time()
        model_fill = load_model(fill_model_path)
        _model_cache.set('fill_model', model_fill)
        logger.info(f"Fill model loaded successfully in {time.time() - start_time:.2f} seconds")
        
        # Load YOLO models, CPU only - Important for Railway
        logger.info("Loading shape detection model...")
        start_time = time.time()
        detector_shape = YOLO(detector_shape_path)
        detector_shape.conf = 0.5  # Match Colab example confidence threshold
        detector_shape.iou = 0.5   
        detector_shape.max_det = 15
        
        # Check for data.yaml file
        shape_yaml_path = str(shape_path / "data.yaml")
        if os.path.exists(shape_yaml_path):
            logger.info(f"Found shape data.yaml file, setting config")
            detector_shape.yaml = shape_yaml_path
        
        detector_shape.to("cpu")  # Ensure CPU usage for Railway compatibility
        _model_cache.set('detector_shape', detector_shape)
        logger.info(f"Shape detection model loaded successfully in {time.time() - start_time:.2f} seconds")
        
        logger.info("Loading card detection model...")
        start_time = time.time()
        detector_card = YOLO(detector_card_path)
        detector_card.conf = 0.5  # Match Colab example confidence threshold
        detector_card.iou = 0.5
        detector_card.max_det = 20
        
        # Check for data.yaml file
        card_yaml_path = str(card_path / "data.yaml")
        if os.path.exists(card_yaml_path):
            logger.info(f"Found card data.yaml file, setting config")
            detector_card.yaml = card_yaml_path
            
        detector_card.to("cpu")  # Ensure CPU usage for Railway compatibility
        _model_cache.set('detector_card', detector_card)
        logger.info(f"Card detection model loaded successfully in {time.time() - start_time:.2f} seconds")
        
        logger.info(f"All models loaded successfully")
        return model_shape, model_fill, detector_card, detector_shape
    
    except ModelLoadError:
        raise
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}", exc_info=True)
        raise ModelLoadError(f"Failed to load models: {str(e)}")


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

def detect_cards(board_img, card_detector):
    """Detect card bounding boxes using YOLO with CPU optimization."""
    h, w = board_img.shape[:2]
    max_dim = 1200
    scale_factor = 1.0
    
    if max(h, w) > max_dim:
        scale_factor = max_dim / max(h, w)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        board_img_small = cv2.resize(board_img, (new_w, new_h))
        result = card_detector(board_img_small)
        boxes = result[0].boxes.xyxy.cpu().numpy() / scale_factor
    else:
        result = card_detector(board_img)
        boxes = result[0].boxes.xyxy.cpu().numpy()
    
    boxes = boxes.astype(int)
    confs = result[0].boxes.conf.cpu().numpy()
    
    good_boxes = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if x1 >= 0 and y1 >= 0 and x2 < w and y2 < h and x2 > x1 and y2 > y1:
            card_area = (x2 - x1) * (y2 - y1)
            image_area = w * h
            if card_area > 0.005 * image_area:  # card must be >=0.5% of the image
                good_boxes.append((board_img[y1:y2, x1:x2], [x1, y1, x2, y2], confs[i]))
    
    logger.info(f"Detected {len(good_boxes)} valid cards")
    return sorted(good_boxes, key=lambda x: x[2], reverse=True)

def predict_card_features(card_img, shape_detector, fill_model, shape_model, card_box):
    """
    Predict features (count, color, fill, shape) for a single card.
    Uses YOLO for shape detection + two Keras models for shape/fill classification.
    """
    c_h, c_w = card_img.shape[:2]
    resized = False
    
    if c_h > 300 or c_w > 300:
        scale = min(300 / c_h, 300 / c_w)
        new_w, new_h = int(c_w * scale), int(c_h * scale)
        card_img_small = cv2.resize(card_img, (new_w, new_h))
        resized = True
        scale_factor = 1 / scale
    else:
        card_img_small = card_img
        scale_factor = 1.0
    
    # Mild contrast enhancement
    adjusted_img = card_img_small.copy()
    lab = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    adjusted_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    # Shape detection
    try:
        shape_detections = shape_detector(adjusted_img)
    except Exception as e:
        logger.error(f"Error detecting shapes on card: {e}")
        return {'count': 0, 'color': 'unknown', 'fill': 'unknown', 'shape': 'unknown', 'box': card_box}
        
    small_card_area = adjusted_img.shape[1] * adjusted_img.shape[0]
    shape_boxes = []
    shape_scores = []
    
    for i, coords in enumerate(shape_detections[0].boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = coords.astype(int)
        shape_area = (x2 - x1) * (y2 - y1)
        
        min_shape_area = 0.02 * small_card_area
        max_shape_area = 0.25 * small_card_area
        
        if min_shape_area < shape_area < max_shape_area:
            conf_score = float(shape_detections[0].boxes.conf.cpu().numpy()[i])
            if resized:
                x1 = int(x1 * scale_factor)
                y1 = int(y1 * scale_factor)
                x2 = int(x2 * scale_factor)
                y2 = int(y2 * scale_factor)
            shape_boxes.append([x1, y1, x2, y2])
            shape_scores.append(conf_score)
    
    if not shape_boxes:
        logger.warning(f"No shapes detected in card at {card_box}")
        return {'count': 0, 'color': 'unknown', 'fill': 'unknown', 'shape': 'unknown', 'box': card_box}
    
    fill_input_size = fill_model.input_shape[1:3]
    shape_input_size = shape_model.input_shape[1:3]
    fill_imgs, shape_imgs, color_candidates = [], [], []
    
    for sb in shape_boxes:
        sx1, sy1, sx2, sy2 = sb
        sy1 = max(0, sy1)
        sy2 = min(card_img.shape[0], sy2)
        sx1 = max(0, sx1)
        sx2 = min(card_img.shape[1], sx2)
        
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        
        shape_crop = card_img[sy1:sy2, sx1:sx2]
        if shape_crop.size == 0:
            continue
        
        fill_imgs.append(cv2.resize(shape_crop, fill_input_size) / 255.0)
        shape_imgs.append(cv2.resize(shape_crop, shape_input_size) / 255.0)
        color_candidates.append(predict_color(shape_crop))
    
    if not fill_imgs:
        return {'count': 0, 'color': 'unknown', 'fill': 'unknown', 'shape': 'unknown', 'box': card_box}
    
    batch_size = min(len(fill_imgs), 4)
    try:
        with tf.device('/cpu:0'):
            fill_preds = fill_model.predict(np.array(fill_imgs), batch_size=batch_size, verbose=0)
            shape_preds = shape_model.predict(np.array(shape_imgs), batch_size=batch_size, verbose=0)
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        return {'count': 0, 'color': 'unknown', 'fill': 'unknown', 'shape': 'unknown', 'box': card_box}
    
    fill_labels = ['empty', 'full', 'striped']
    shape_labels = ['diamond', 'oval', 'squiggle']
    
    fill_result = [fill_labels[np.argmax(fp)] for fp in fill_preds]
    shape_result = [shape_labels[np.argmax(sp)] for sp in shape_preds]
    
    shape_count = len(shape_boxes)
    color_candidates = [c for c in color_candidates if c != 'unknown']
    
    if not color_candidates:
        color = 'unknown'
    else:
        color = max(set(color_candidates), key=color_candidates.count)
        
    fill = max(set(fill_result), key=fill_result.count)
    shape = max(set(shape_result), key=shape_result.count)
    
    # Translate fill to standard terms
    fill_mapping = {
        'empty': 'outline',
        'full': 'solid',
        'striped': 'striped'
    }
    
    return {
        'count': shape_count,
        'color': color,
        'fill': fill_mapping.get(fill, fill),
        'shape': shape,
        'box': card_box
    }

def classify_cards_on_board(board_img, card_detector, shape_detector, fill_model, shape_model):
    """Detect and classify all cards on the board."""
    card_rows = []
    try:
        card_data = detect_cards(board_img, card_detector)
    except Exception as e:
        logger.error(f"Error detecting cards on board: {e}")
        return pd.DataFrame()
    
    if not card_data:
        logger.warning("No cards detected on board")
        return pd.DataFrame()
    
    for card_img, box, conf in card_data:
        try:
            card_feats = predict_card_features(card_img, shape_detector, fill_model, shape_model, box)
            valid_features = True
            for key in ['count', 'color', 'fill', 'shape']:
                if card_feats[key] == 'unknown' or card_feats[key] == 0:
                    valid_features = False
                    break
            if valid_features:
                card_rows.append({
                    "Count": card_feats['count'],
                    "Color": card_feats['color'],
                    "Fill": card_feats['fill'],
                    "Shape": card_feats['shape'],
                    "Coordinates": card_feats['box'],
                    "Confidence": conf
                })
        except Exception as e:
            logger.error(f"Error processing card: {e}")
            continue
    
    if len(card_rows) < 3:
        logger.warning(f"Not enough valid cards detected: {len(card_rows)}")
    else:
        logger.info(f"Successfully classified {len(card_rows)} cards")
        
    return pd.DataFrame(card_rows)

def valid_set(cards):
    """Check if 3 cards form a valid SET."""
    for feature in ["Count", "Color", "Fill", "Shape"]:
        values = set(card[feature] for card in cards)
        if len(values) not in (1, 3):
            return False
    return True

def locate_all_sets(cards_df):
    """Find all possible SETs from the cards."""
    found_sets = []
    if len(cards_df) < 3:
        logger.warning(f"Not enough cards to form a SET: {len(cards_df)}")
        return found_sets
        
    valid_cards = cards_df[
        (cards_df['Count'] > 0) &
        (cards_df['Color'] != 'unknown') &
        (cards_df['Fill'] != 'unknown') &
        (cards_df['Shape'] != 'unknown')
    ]
    
    if len(valid_cards) < 3:
        logger.warning(f"Not enough valid cards after filtering: {len(valid_cards)}")
        return found_sets
    
    for combo in combinations(valid_cards.iterrows(), 3):
        cards = [c[1] for c in combo]
        if valid_set(cards):
            found_sets.append({
                'set_indices': [c[0] for c in combo],
                'cards': [{
                    f: card[f] for f in ['Count', 'Color', 'Fill', 'Shape', 'Coordinates']
                } for card in cards]
            })
    
    logger.info(f"Found {len(found_sets)} valid SETs")
    return found_sets

def draw_set_indicators(img, sets):
    """
    Draw bounding boxes around each card in the detected SETs. 
    Removed the connecting lines to focus only on boxes.
    """
    result = img.copy()
    
    # Define a color palette with higher contrast
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
        (0, 255, 128)   # Teal
    ]
    
    for idx, set_info in enumerate(sets):
        color = colors[idx % len(colors)]
        
        # Draw rectangles around each card (with a shadow effect)
        for card in set_info['cards']:
            x1, y1, x2, y2 = card['Coordinates']
            
            shadow_offset = 2
            # Draw a black shadow rectangle
            cv2.rectangle(result, (x1 - shadow_offset, y1 - shadow_offset), 
                          (x2 + shadow_offset, y2 + shadow_offset), (0, 0, 0), 5)
            # Draw the main rectangle in color
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
            
            # Label the set number for clarity
            label = f"Set {idx + 1}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(
                result,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0] + 10, y1),
                color,
                -1
            )
            cv2.putText(result, label, (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    return result

def identify_sets(image):
    """
    Complete pipeline to find SETs in an image:
      1. Load models (with caching).
      2. Possibly resize and denoise input for performance.
      3. Correct orientation if needed.
      4. Detect cards, then classify each one.
      5. Locate all valid SETs among them.
      6. Draw bounding boxes around each SET (no connecting lines).
      7. Return the found sets and the annotated image.
    """
    start_time = time.time()
    logger.info(f"Starting SET detection on image of shape {image.shape}")
    
    try:
        # Load models with error handling
        try:
            model_shape, model_fill, detector_card, detector_shape = load_models()
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return [], image
        
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
