from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import cv2
import numpy as np
import logging
import io
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

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# In-memory session storage
current_sessions = {}

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/detect_sets', methods=['POST'])
def detect_sets():
    """Handles image uploads and detects SETs."""
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
            
            # Run SET detection
            found_sets, annotated_img = identify_sets(img)
            
            # Encode images to send back to client
            _, original_buffer = cv2.imencode('.jpg', img)
            _, result_buffer = cv2.imencode('.jpg', annotated_img)
            
            # Store session results temporarily
            current_sessions[session_id] = {
                'original': original_buffer,
                'result': result_buffer
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
    
    img_buffer = current_sessions[session_id][image_type]
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
