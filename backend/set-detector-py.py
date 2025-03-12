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

# Configure logging
logger = logging.getLogger(__name__)

# Cached models
_models = None

# Model loading functions with caching
def load_models():
    global _models
    if _models is not None:
        return _models
    
    base_dir = Path("models")
    char_path = base_dir / "Characteristics" / "11022025"
    shape_path = base_dir / "Shape" / "15052024" 
    card_path = base_dir / "Card" / "16042024"
    
    # Ensure model directories exist
    os.makedirs(char_path, exist_ok=True)
    os.makedirs(shape_path, exist_ok=True)
    os.makedirs(card_path, exist_ok=True)
    
    try:
        # Check if model files exist
        shape_model_path = str(char_path / "shape_model.keras")
        fill_model_path = str(char_path / "fill_model.keras")
        detector_shape_path = str(shape_path / "best.pt")
        detector_card_path = str(card_path / "best.pt")
        
        missing_files = []
        for path in [shape_model_path, fill_model_path, detector_shape_path, detector_card_path]:
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")
        
        # Load classification models
        logger.info("Loading shape classification model...")
        model_shape = load_model(shape_model_path)
        
        logger.info("Loading fill classification model...")
        model_fill = load_model(fill_model_path)
        
        # Load detection models
        logger.info("Loading shape detection model...")
        detector_shape = YOLO(detector_shape_path)
        detector_shape.conf = 0.5
        
        logger.info("Loading card detection model...")
        detector_card = YOLO(detector_card_path)
        detector_card.conf = 0.5
        
        # Use GPU if available
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU acceleration")
            detector_card.to("cuda")
            detector_shape.to("cuda")
        else:
            logger.info("CUDA not available, using CPU")
        
        _models = (model_shape, model_fill, detector_card, detector_shape)
        logger.info("All models loaded successfully")
        return _models
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to load models: {str(e)}")

# Core SET detection functions
def correct_orientation(board_image, card_detector):
    """Rotate image if cards are vertical"""
    detection = card_detector(board_image)
    boxes = detection[0].boxes.xyxy.cpu().numpy().astype(int)
    if boxes.size == 0: return board_image, False
    
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    
    return (cv2.rotate(board_image, cv2.ROTATE_90_CLOCKWISE), True) if np.mean(heights) > np.mean(widths) else (board_image, False)

def restore_orientation(img, was_rotated):
    """Restore original orientation if needed"""
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) if was_rotated else img

def predict_color(img_bgr):
    """Classify color using HSV thresholds"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
    mask_purple = cv2.inRange(hsv, np.array([120, 50, 50]), np.array([160, 255, 255]))
    
    # Red wraps around hue=0
    mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    counts = {"green": cv2.countNonZero(mask_green), "purple": cv2.countNonZero(mask_purple), 
              "red": cv2.countNonZero(mask_red)}
    return max(counts, key=counts.get)

def detect_cards(board_img, card_detector):
    """Detect card bounding boxes using YOLO"""
    result = card_detector(board_img)
    boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
    return [(board_img[y1:y2, x1:x2], [x1, y1, x2, y2]) for x1, y1, x2, y2 in boxes]

def predict_card_features(card_img, shape_detector, fill_model, shape_model, card_box):
    """Predict features (count, color, fill, shape) for a card"""
    # Detect shapes on card
    shape_detections = shape_detector(card_img)
    c_h, c_w = card_img.shape[:2]
    card_area = c_w * c_h
    
    # Filter shape detections
    shape_boxes = []
    for coords in shape_detections[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = coords.astype(int)
        if (x2 - x1) * (y2 - y1) > 0.03 * card_area:
            shape_boxes.append([x1, y1, x2, y2])
    
    if not shape_boxes:
        return {'count': 0, 'color': 'unknown', 'fill': 'unknown', 'shape': 'unknown', 'box': card_box}
    
    # Process each shape for classification
    fill_input_size = fill_model.input_shape[1:3]
    shape_input_size = shape_model.input_shape[1:3]
    fill_imgs, shape_imgs, color_candidates = [], [], []
    
    for sb in shape_boxes:
        sx1, sy1, sx2, sy2 = sb
        shape_crop = card_img[sy1:sy2, sx1:sx2]
        fill_imgs.append(cv2.resize(shape_crop, fill_input_size) / 255.0)
        shape_imgs.append(cv2.resize(shape_crop, shape_input_size) / 255.0)
        color_candidates.append(predict_color(shape_crop))
    
    # Classify shapes and fills
    fill_preds = fill_model.predict(np.array(fill_imgs), batch_size=len(fill_imgs), verbose=0)
    shape_preds = shape_model.predict(np.array(shape_imgs), batch_size=len(shape_imgs), verbose=0)
    
    fill_labels = ['empty', 'full', 'striped']
    shape_labels = ['diamond', 'oval', 'squiggle']
    
    fill_result = [fill_labels[np.argmax(fp)] for fp in fill_preds]
    shape_result = [shape_labels[np.argmax(sp)] for sp in shape_preds]
    
    return {
        'count': len(shape_boxes),
        'color': max(set(color_candidates), key=color_candidates.count),
        'fill': max(set(fill_result), key=fill_result.count),
        'shape': max(set(shape_result), key=shape_result.count),
        'box': card_box
    }

def classify_cards_on_board(board_img, card_detector, shape_detector, fill_model, shape_model):
    """Detect and classify all cards on the board"""
    card_rows = []
    for card_img, box in detect_cards(board_img, card_detector):
        card_feats = predict_card_features(card_img, shape_detector, fill_model, shape_model, box)
        card_rows.append({
            "Count": card_feats['count'],
            "Color": card_feats['color'],
            "Fill": card_feats['fill'],
            "Shape": card_feats['shape'],
            "Coordinates": card_feats['box']
        })
    return pd.DataFrame(card_rows)

def valid_set(cards):
    """Check if 3 cards form a valid SET"""
    for feature in ["Count", "Color", "Fill", "Shape"]:
        # Each feature must be all same or all different
        if len(set(card[feature] for card in cards)) not in (1, 3):
            return False
    return True

def locate_all_sets(cards_df):
    """Find all possible SETs from the cards"""
    found_sets = []
    for combo in combinations(cards_df.iterrows(), 3):
        cards = [c[1] for c in combo]
        if valid_set(cards):
            found_sets.append({
                'set_indices': [c[0] for c in combo],
                'cards': [{f: card[f] for f in ['Count', 'Color', 'Fill', 'Shape', 'Coordinates']} 
                          for card in cards]
            })
    return found_sets

def draw_set_indicators(img, sets):
    """Draw SET indicators on the image"""
    result = img.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
    
    for idx, set_info in enumerate(sets):
        color = colors[idx % len(colors)]
        for card in set_info['cards']:
            x1, y1, x2, y2 = card['Coordinates']
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
            
    return result

def identify_sets(image):
    """Complete pipeline to find SETs in an image"""
    # Load models
    model_shape, model_fill, detector_card, detector_shape = load_models()
    
    # Correct orientation
    processed, was_rotated = correct_orientation(image, detector_card)
    
    # Detect cards and find sets
    df_cards = classify_cards_on_board(processed, detector_card, detector_shape, model_fill, model_shape)
    found_sets = locate_all_sets(df_cards)
    
    # Draw results
    if found_sets:
        annotated = draw_set_indicators(processed.copy(), found_sets)
        final_image = restore_orientation(annotated, was_rotated)
        return found_sets, final_image
    else:
        return [], restore_orientation(processed, was_rotated)
