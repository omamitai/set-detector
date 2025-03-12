from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import cv2
import numpy as np
import logging
import io
import time
import gc  # Garbage collection
from werkzeug.utils import secure_filename
from set_detector import identify_sets

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Get environment variables
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', '2'))
app.logger.info(f"Configured with MAX_WORKERS={MAX_WORKERS}")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# In-memory session storage with TTL
current_sessions = {}
SESSION_TTL = 10 * 60  # 10 minutes TTL for sessions
CLEANUP_INTERVAL = 60  # Check for expired sessions every minute
last_cleanup = time.time()

def cleanup_old_sessions():
    """Remove session data older than TTL"""
    global last_cleanup
    now = time.time()
    
    # Only run cleanup periodically to reduce overhead
    if now - last_cleanup < CLEANUP_INTERVAL:
        return
        
    app.logger.info("Running session cleanup")
    last_cleanup = now
    expired_sessions = []
    
    for session_id, session_data in current_sessions.items():
        if now - session_data.get('timestamp', 0) > SESSION_TTL:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        app.logger.info(f"Removing expired session: {session_id}")
        current_sessions.pop(session_id, None)
        
    # Force garbage collection
    gc.collect()
    app.logger.info(f"Cleanup finished. Removed {len(expired_sessions)} sessions. {len(current_sessions)} active sessions remain.")

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/detect_sets', methods=['POST'])
def detect_sets():
    """Handles image uploads and detects SETs."""
    # Cleanup old sessions
    cleanup_old_sessions()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        session_id = str(uuid.uuid4())
        
        try:
            # Read image into memory
            file_data = file.read()
            nparr = np.frombuffer(file_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                app.logger.error("Failed to decode image")
                return jsonify({'error': 'Could not process uploaded image'}), 400
                
            app.logger.info(f"Image uploaded with shape {img.shape}")
            
            # Resize large images for CPU efficiency
            max_dimension = 1500  # Maximum dimension for processing
            h, w = img.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
                app.logger.info(f"Resized image to {img.shape} for processing efficiency")
            
            # Run SET detection
            found_sets, annotated_img = identify_sets(img)
            
            # Encode images with optimal quality for web
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # 85% quality is good balance
            _, original_buffer = cv2.imencode('.jpg', img, encode_params)
            _, result_buffer = cv2.imencode('.jpg', annotated_img, encode_params)
            
            # Free memory
            del img
            del annotated_img
            gc.collect()
            
            # Store session results temporarily
            current_sessions[session_id] = {
                'original': original_buffer,
                'result': result_buffer,
                'timestamp': time.time()
            }
            
            # Format response data
            detected_sets = []
            for set_info in found_sets:
                cards = [f"{card['Count']} {card['Fill']} {card['Color']} {card['Shape']}" 
                         for card in set_info['cards']]
                
                coordinates = [{'x': (box[0] + box[2]) // 2, 'y': (box[1] + box[3]) // 2} 
                               for box in [card['Coordinates'] for card in set_info['cards']]]
                
                detected_sets.append({
                    'cards': cards,
                    'coordinates': coordinates
                })
            
            app.logger.info(f"Found {len(detected_sets)} SETs in image")
            
            return jsonify({
                'session_id': session_id,
                'original_image_url': f"/api/images/{session_id}/original",
                'processed_image_url': f"/api/images/{session_id}/result",
                'detected_sets': detected_sets
            })
        
        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}", exc_info=True)
            return jsonify({'error': 'Internal server error'}), 500
    
    return jsonify({'error': 'Invalid file type. Only PNG and JPEG are supported.'}), 400

@app.route('/api/images/<session_id>/<image_type>')
def get_image(session_id, image_type):
    """Returns processed images from memory."""
    if session_id not in current_sessions:
        return jsonify({'error': 'Image not found or session expired'}), 404
        
    if image_type not in ['original', 'result']:
        return jsonify({'error': 'Invalid image type'}), 400
    
    # Update session timestamp to keep it alive
    current_sessions[session_id]['timestamp'] = time.time()
    
    img_buffer = current_sessions[session_id][image_type]
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring."""
    # Return memory usage stats
    memory_info = {
        'active_sessions': len(current_sessions),
    }
    return jsonify({
        'status': 'healthy',
        'memory': memory_info
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
