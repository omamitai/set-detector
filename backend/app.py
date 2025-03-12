from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from set_detector import identify_sets

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/detect_sets', methods=['POST'])
def detect_sets():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        base, ext = os.path.splitext(filename)
        unique_filename = f"{unique_id}{ext}"
        
        # Save original image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        try:
            # Read and process image
            img = cv2.imread(file_path)
            if img is None:
                app.logger.error(f"Failed to read image at {file_path}")
                return jsonify({'error': 'Could not read uploaded image'}), 400
                
            app.logger.info(f"Successfully read image with shape {img.shape}")
            
            # Run SET detection
            found_sets, annotated_img = identify_sets(img)
            
            # Save annotated image
            result_filename = f"{unique_id}_result{ext}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, annotated_img)
            
            # Prepare response
            detected_sets = []
            for set_info in found_sets:
                cards = []
                for card in set_info['cards']:
                    card_str = f"{card['Count']} {card['Fill']} {card['Color']} {card['Shape']}"
                    cards.append(card_str)
                    
                coordinates = [{'x': (box[0] + box[2]) // 2, 'y': (box[1] + box[3]) // 2} 
                              for box in [card['Coordinates'] for card in set_info['cards']]]
                
                detected_sets.append({
                    'cards': cards,
                    'coordinates': coordinates
                })
            
            # Return results
            return jsonify({
                'original_image_url': f"/api/images/{unique_filename}",
                'processed_image_url': f"/api/images/{result_filename}",
                'detected_sets': detected_sets
            })
            
        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}", exc_info=True)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload a PNG or JPEG image.'}), 400

@app.route('/api/images/<filename>')
def get_image(filename):
    # Check if it's a result image
    if filename.find('_result') > 0:
        return send_from_directory(app.config['RESULT_FOLDER'], filename)
    # Otherwise, it's an original image
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/health')
def health_check():
    """Health check endpoint for container monitoring"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
