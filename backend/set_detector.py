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
    # Get available CPU count for efficient threading
    import multiprocessing
    CPU_COUNT = multiprocessing.cpu_count()
    THREAD_COUNT = min(2, CPU_COUNT)
    
    logger.info(f"Configuring TensorFlow with {THREAD_COUNT} threads (detected {CPU_COUNT} CPUs)")
    
    # Set thread count based on available CPUs but keep it modest for EC2 t-series
    tf.config.threading.set_intra_op_parallelism_threads(THREAD_COUNT)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Additional optimizations
    tf.compat.v1.disable_eager_execution()
    
    # Memory growth for GPU if available
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
                # Update timestamp on access
                item['timestamp'] = time.time()
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
        if now - self._last_cleanup < 600:  # Run cleanup every 10 minutes
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

# Create model cache as module-level variable to ensure sharing between workers
_model_cache = ModelCache()

def load_models():
    """Load models with caching and proper error handling"""
    # Check cache first
    models = (_model_cache.get('shape_model'), 
              _model_cache.get('fill_model'),
              _model_cache.get('detector_card'),
              _model_cache.get('detector_shape'))
    
    if all(models):
        logger.debug("Using cached models")
        return models
    
    logger.info("Loading models from disk")
    
    base_dir = Path("models")
    char_path = base_dir / "Characteristics" / "11022025"
    shape_path = base_dir / "Shape" / "15052024" 
    card_path = base_dir / "Card" / "16042024"
    
    # Verify model directories exist
    for path, name in [(char_path, 'Characteristics'), 
                       (shape_path, 'Shape'), 
                       (card_path, 'Card')]:
        if not path.exists():
            error_msg = f"Model directory {name} not found at {path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    try:
        # Define model paths
        shape_model_path = str(char_path / "shape_model.keras")
        fill_model_path = str(char_path / "fill_model.keras")
        detector_shape_path = str(shape_path / "best.pt")
        detector_card_path = str(card_path / "best.pt")
        
        # Check if model files exist
        missing_files = []
        for path in [shape_model_path, fill_model_path, detector_shape_path, detector_card_path]:
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            error_msg = f"Missing model files: {', '.join(missing_files)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load classification models
        start_time = time.time()
        logger.info("Loading shape classification model...")
        model_shape = load_model(shape_model_path)
        _model_cache.set('shape_model', model_shape)
        
        logger.info("Loading fill classification model...")
        model_fill = load_model(fill_model_path)
        _model_cache.set('fill_model', model_fill)
        
        # Load YOLO models with CPU optimization
        logger.info("Loading shape detection model...")
        detector_shape = YOLO(detector_shape_path)
        # CPU optimization settings
        detector_shape.conf = 0.65  # Higher confidence threshold for better accuracy and speed
        detector_shape.iou = 0.5   
        detector_shape.max_det = 15  # Increased from 10 for better detection
        detector_shape.to("cpu")
        _model_cache.set('detector_shape', detector_shape)
        
        logger.info("Loading card detection model...")
        detector_card = YOLO(detector_card_path)
        detector_card.conf = 0.65
        detector_card.iou = 0.5
        detector_card.max_det = 20
        detector_card.to("cpu")
        _model_cache.set('detector_card', detector_card)
        
        logger.info(f"All models loaded successfully in {time.time() - start_time:.2f} seconds")
        return model_shape, model_fill, detector_card, detector_shape
    
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to load models: {str(e)}")

# Core SET detection functions with optimizations
def correct_orientation(board_image, card_detector):
    """Rotate image if cards are vertical, with optimized processing"""
    # Resize large images for faster detection
    h, w = board_image.shape[:2]
    max_dim = 1200
    
    # Optimize by only detecting on a smaller version
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
        logger.info(f"Rotating image 90° clockwise (detected vertical layout)")
        return cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE), True
    else:
        return board_image, False

def restore_orientation(img, was_rotated):
    """Restore original orientation if needed"""
    if was_rotated:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def predict_color(img_bgr):
    """Classify color using HSV thresholds with enhanced accuracy"""
    # Convert to HSV for better color discrimination
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Enhanced color thresholds with better tolerance
    mask_green = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([85, 255, 255]))
    mask_purple = cv2.inRange(hsv, np.array([115, 40, 40]), np.array([165, 255, 255]))
    
    # Red wraps around hue=0, capture both ends of the spectrum
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Count and normalize by mask size for better accuracy
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    
    if total_pixels == 0:  # Avoid division by zero
        return "unknown"
        
    counts = {
        "green": cv2.countNonZero(mask_green) / total_pixels,
        "purple": cv2.countNonZero(mask_purple) / total_pixels, 
        "red": cv2.countNonZero(mask_red) / total_pixels
    }
    
    # Only classify if we have a clear signal
    max_color = max(counts, key=counts.get)
    if counts[max_color] < 0.05:  # Need at least 5% of pixels matching
        return "unknown"
        
    return max_color

def detect_cards(board_img, card_detector):
    """Detect card bounding boxes using YOLO with CPU optimization"""
    # Image resolution management for efficiency
    h, w = board_img.shape[:2]
    max_dim = 1200
    scale_factor = 1.0
    
    # Resize large images for faster detection
    if max(h, w) > max_dim:
        scale_factor = max_dim / max(h, w)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        board_img_small = cv2.resize(board_img, (new_w, new_h))
        result = card_detector(board_img_small)
        
        # Scale boxes back
        boxes = result[0].boxes.xyxy.cpu().numpy()
        boxes = boxes / scale_factor
    else:
        result = card_detector(board_img)
        boxes = result[0].boxes.xyxy.cpu().numpy()
    
    # Convert to integers and filter by confidence
    boxes = boxes.astype(int)
    
    # Get confidence scores
    confs = result[0].boxes.conf.cpu().numpy()
    
    # Only keep boxes with good confidence
    good_boxes = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Validate box coordinates
        if x1 >= 0 and y1 >= 0 and x2 < w and y2 < h and x2 > x1 and y2 > y1:
            # Additional check for minimum card size (avoid false positives)
            card_area = (x2 - x1) * (y2 - y1)
            image_area = w * h
            if card_area > 0.005 * image_area:  # Card must be at least 0.5% of image
                good_boxes.append((board_img[y1:y2, x1:x2], [x1, y1, x2, y2], confs[i]))
    
    logger.info(f"Detected {len(good_boxes)} valid cards")
    
    # Sort by confidence (for display purposes - high confidence cards first)
    return sorted(good_boxes, key=lambda x: x[2], reverse=True)

def predict_card_features(card_img, shape_detector, fill_model, shape_model, card_box):
    """Predict features (count, color, fill, shape) for a card with optimizations"""
    # Resize large card images for faster processing
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
    
    # Color adjustment to improve detection in various lighting
    adjusted_img = card_img_small.copy()
    
    # Apply mild contrast enhancement
    lab = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    adjusted_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    # Run shape detection on processed image
    shape_detections = shape_detector(adjusted_img)
    small_card_area = adjusted_img.shape[1] * adjusted_img.shape[0]
    
    # Filter shape detections
    shape_boxes = []
    shape_scores = []
    
    for i, coords in enumerate(shape_detections[0].boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = coords.astype(int)
        shape_area = (x2 - x1) * (y2 - y1)
        
        # Adjust shape area threshold based on card size
        min_shape_area = 0.02 * small_card_area
        max_shape_area = 0.25 * small_card_area
        
        if min_shape_area < shape_area < max_shape_area:
            conf_score = float(shape_detections[0].boxes.conf.cpu().numpy()[i])
            
            if resized:
                # Scale coordinates back to original size
                x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
                
            shape_boxes.append([x1, y1, x2, y2])
            shape_scores.append(conf_score)
    
    # Fail case handling
    if not shape_boxes:
        logger.warning(f"No shapes detected in card at {card_box}")
        return {'count': 0, 'color': 'unknown', 'fill': 'unknown', 'shape': 'unknown', 'box': card_box}
    
    # Process each shape for classification
    fill_input_size = fill_model.input_shape[1:3]
    shape_input_size = shape_model.input_shape[1:3]
    fill_imgs, shape_imgs, color_candidates = [], [], []
    
    for sb in shape_boxes:
        sx1, sy1, sx2, sy2 = sb
        # Boundary check to prevent crashes
        sy1 = max(0, sy1)
        sy2 = min(card_img.shape[0], sy2)
        sx1 = max(0, sx1)
        sx2 = min(card_img.shape[1], sx2)
        
        # Skip invalid boxes
        if sx2 <= sx1 or sy2 <= sy1:
            continue
            
        shape_crop = card_img[sy1:sy2, sx1:sx2]
        
        # Skip empty crops
        if shape_crop.size == 0:
            continue
            
        fill_imgs.append(cv2.resize(shape_crop, fill_input_size) / 255.0)
        shape_imgs.append(cv2.resize(shape_crop, shape_input_size) / 255.0)
        color_candidates.append(predict_color(shape_crop))
    
    # In case all crops were invalid
    if not fill_imgs:
        return {'count': 0, 'color': 'unknown', 'fill': 'unknown', 'shape': 'unknown', 'box': card_box}
    
    # Classify shapes and fills with optimized batch sizing
    batch_size = min(len(fill_imgs), 4)  # Small batch size for CPU efficiency
    
    # Batch prediction for better performance
    with tf.device('/cpu:0'):  # Force CPU for inference
        fill_preds = fill_model.predict(np.array(fill_imgs), batch_size=batch_size, verbose=0)
        shape_preds = shape_model.predict(np.array(shape_imgs), batch_size=batch_size, verbose=0)
    
    fill_labels = ['empty', 'full', 'striped']
    shape_labels = ['diamond', 'oval', 'squiggle']
    
    fill_result = [fill_labels[np.argmax(fp)] for fp in fill_preds]
    shape_result = [shape_labels[np.argmax(sp)] for sp in shape_preds]
    
    # Determine count from shape detection results
    shape_count = len(shape_boxes)
    
    # Filtering out unknowns for better accuracy
    color_candidates = [c for c in color_candidates if c != 'unknown']
    
    # Determine most frequent values
    if not color_candidates:
        color = 'unknown'
    else:
        color = max(set(color_candidates), key=color_candidates.count)
        
    fill = max(set(fill_result), key=fill_result.count)
    shape = max(set(shape_result), key=shape_result.count)
    
    return {
        'count': shape_count,
        'color': color,
        'fill': fill,
        'shape': shape,
        'box': card_box
    }

def classify_cards_on_board(board_img, card_detector, shape_detector, fill_model, shape_model):
    """Detect and classify all cards on the board with confidence tracking"""
    card_rows = []
    
    # Get card images and boxes with confidence scores
    card_data = detect_cards(board_img, card_detector)
    
    if not card_data:
        logger.warning("No cards detected on board")
        return pd.DataFrame()
    
    # Process each card
    for card_img, box, conf in card_data:
        card_feats = predict_card_features(card_img, shape_detector, fill_model, shape_model, box)
        
        # Add confidence score and validate features
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
    
    # If we have too few cards, log a warning
    if len(card_rows) < 3:
        logger.warning(f"Not enough valid cards detected: {len(card_rows)}")
    else:
        logger.info(f"Successfully classified {len(card_rows)} cards")
        
    return pd.DataFrame(card_rows)

def valid_set(cards):
    """Check if 3 cards form a valid SET"""
    for feature in ["Count", "Color", "Fill", "Shape"]:
        values = set(card[feature] for card in cards)
        # Each feature must be all same or all different
        if len(values) not in (1, 3):
            return False
    return True

def locate_all_sets(cards_df):
    """Find all possible SETs from the cards"""
    found_sets = []
    
    if len(cards_df) < 3:
        logger.warning(f"Not enough cards to form a SET: {len(cards_df)} cards")
        return found_sets
        
    # Filter out cards with unknown attributes
    valid_cards = cards_df[
        (cards_df['Count'] > 0) & 
        (cards_df['Color'] != 'unknown') & 
        (cards_df['Fill'] != 'unknown') & 
        (cards_df['Shape'] != 'unknown')
    ]
    
    if len(valid_cards) < 3:
        logger.warning(f"Not enough valid cards after filtering: {len(valid_cards)} cards")
        return found_sets
    
    # Find all valid SETs
    for combo in combinations(valid_cards.iterrows(), 3):
        cards = [c[1] for c in combo]
        if valid_set(cards):
            found_sets.append({
                'set_indices': [c[0] for c in combo],
                'cards': [{f: card[f] for f in ['Count', 'Color', 'Fill', 'Shape', 'Coordinates']} 
                          for card in cards]
            })
            
    logger.info(f"Found {len(found_sets)} valid SETs")
    return found_sets

def draw_set_indicators(img, sets):
    """Draw SET indicators on the image with enhanced visualization"""
    result = img.copy()
    
    # Define a better color palette with higher contrast
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
        
        # Draw connecting lines between cards in the SET
        centers = []
        for card in set_info['cards']:
            x1, y1, x2, y2 = card['Coordinates']
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            centers.append((center_x, center_y))
        
        # Draw triangle connecting the three cards
        if len(centers) == 3:
            cv2.line(result, centers[0], centers[1], color, 2)
            cv2.line(result, centers[1], centers[2], color, 2)
            cv2.line(result, centers[2], centers[0], color, 2)
        
        # Draw rectangles around each card
        for card in set_info['cards']:
            x1, y1, x2, y2 = card['Coordinates']
            
            # Draw box with shadow effect for better visibility
            shadow_offset = 2
            cv2.rectangle(result, (x1-shadow_offset, y1-shadow_offset), 
                         (x2+shadow_offset, y2+shadow_offset), (0, 0, 0), 5)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
            
            # Add set number for reference
            label = f"Set {idx+1}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color, -1)
            cv2.putText(result, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    return result

def identify_sets(image):
    """Complete pipeline to find SETs in an image with enhanced performance and error handling"""
    start_time = time.time()
    logger.info(f"Starting SET detection on image of shape {image.shape}")
    
    try:
        # Load models
        model_shape, model_fill, detector_card, detector_shape = load_models()
        
        # Resize large images for CPU efficiency
        h, w = image.shape[:2]
        max_dim = 1500
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            logger.info(f"Resized input image from {(w, h)} to {(new_w, new_h)} for CPU efficiency")
        
        # Apply mild preprocessing for better detection
        image_enhanced = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        
        # Correct orientation
        processed, was_rotated = correct_orientation(image_enhanced, detector_card)
        
        # Detect cards and find sets
        df_cards = classify_cards_on_board(processed, detector_card, detector_shape, model_fill, model_shape)
        
        # Handle empty results
        if df_cards.empty:
            logger.warning("No valid cards detected in the image")
            return [], image
            
        found_sets = locate_all_sets(df_cards)
        
        # Draw results
        if found_sets:
            annotated = draw_set_indicators(processed.copy(), found_sets)
            final_image = restore_orientation(annotated, was_rotated)
            
            logger.info(f"SET detection complete. Found {len(found_sets)} SETs in {time.time() - start_time:.2f} seconds")
            return found_sets, final_image
        else:
            logger.info(f"No SETs found in the image after {time.time() - start_time:.2f} seconds")
            return [], restore_orientation(processed, was_rotated)
            
    except Exception as e:
        logger.error(f"Error in SET detection: {str(e)}", exc_info=True)
        # Return original image in case of error
        return [], image
