from flask_cors import CORS
import os
import uuid
import cv2
import numpy as np
import logging
import io
import time
import gc  # Garbage collection
import resource
import sys
import json
from werkzeug.utils import secure_filename
from set_detector import identify_sets, load_models, ModelLoadError

# Add this to the top of your app.py file, after imports
from flask import Flask, request, jsonify, send_file, after_this_request

# Then replace your CORS setup with this more robust version
def setup_cors(app):
    """Set up CORS with both middleware and per-request handlers for maximum compatibility"""
    CORS_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
    app.logger.info(f"Configuring CORS with allowed origins: {CORS_ORIGINS}")
    
    # Use Flask-CORS extension
    CORS(app, 
         resources={r"/*": {"origins": "*"}},
         supports_credentials=False,
         methods=["GET", "POST", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
         expose_headers=["Content-Disposition"],
         max_age=86400)  # Cache preflight requests for 24 hours
         
    # Also add before_request and after_request handlers for double protection
    @app.before_request
    def handle_options():
        """Handle OPTIONS requests explicitly for CORS preflight"""
        if request.method == 'OPTIONS':
            response = app.make_default_options_response()
            add_cors_headers(response)
            return response
    
    @app.after_request
    def add_cors_headers(response):
        """Add CORS headers to every response"""
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
        response.headers['Access-Control-Max-Age'] = '86400'  # 24 hours
        return response
    
    return app

# Then use it to initialize your app
app = Flask(__name__)
app = setup_cors(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Get environment variables with Railway-compatible defaults
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', '2'))
PORT = int(os.environ.get('PORT', '5000'))  # Railway provides PORT env var
app.logger.info(f"Configured with MAX_WORKERS={MAX_WORKERS}, PORT={PORT}")


# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/jpg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# In-memory session storage with configurable limits
MAX_SESSIONS = int(os.environ.get('MAX_SESSIONS', '20'))
SESSION_TTL = int(os.environ.get('SESSION_TTL', '300'))  # 5 minutes default
CLEANUP_INTERVAL = int(os.environ.get('CLEANUP_INTERVAL', '60'))  # Check every minute
MAX_MEMORY_PERCENT = int(os.environ.get('MAX_MEMORY_PERCENT', '80'))  # Memory threshold
current_sessions = {}
last_cleanup = time.time()
# Track application startup time
app_startup_time = time.time()

# Try to load models, but don't fail startup if they can't be loaded
# This allows health checks to pass while models initialize
try:
    app.logger.info("Attempting to load models on startup...")
    load_models()
    app.logger.info("Models loaded successfully!")
    models_available = True
except Exception as e:
    app.logger.error(f"Failed to load models on startup: {e}", exc_info=True)
    app.logger.warning("Application will continue without models - health checks will pass")
    models_available = False

def check_memory_pressure():
    """Check if system is under memory pressure with improved reliability"""
    try:
        # Get self memory usage
        mem_info = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Convert to MB (platform-specific)
        if sys.platform == 'darwin':  # macOS
            mem_usage_mb = mem_info / 1024 / 1024
        else:  # Linux and others
            mem_usage_mb = mem_info / 1024
            
        app.logger.debug(f"Memory usage: {mem_usage_mb:.2f}MB")
        
        # Railway has memory limits of 2GB by default in standard tier
        return mem_usage_mb > (int(os.environ.get('MEMORY_LIMIT_MB', '1536')))
            
    except Exception as e:
        app.logger.warning(f"Error checking memory pressure: {e}")
        # Default to False to prevent unnecessary cleanup
        return False

def cleanup_old_sessions(force=False):
    """Remove session data older than TTL or when memory limits are approached"""
    global last_cleanup
    now = time.time()
    
    # Only run cleanup periodically to reduce overhead, unless forced
    if not force and now - last_cleanup < CLEANUP_INTERVAL:
        return
        
    app.logger.info("Running session cleanup")
    last_cleanup = now
    expired_sessions = []
    
    # Check if we're approaching memory limits
    memory_pressure = check_memory_pressure()
    
    for session_id, session_data in current_sessions.items():
        # Remove expired sessions
        if now - session_data.get('timestamp', 0) > SESSION_TTL:
            expired_sessions.append(session_id)
        # If under memory pressure, be more aggressive with cleanup
        elif memory_pressure and len(current_sessions) > MAX_SESSIONS // 2:
            # Sort by age and keep only recent sessions
            sessions_by_age = sorted(
                current_sessions.items(), 
                key=lambda x: x[1].get('timestamp', 0)
            )
            # Keep newest half of sessions
            to_keep = sessions_by_age[-(MAX_SESSIONS//2):]
            keep_ids = [s[0] for s in to_keep]
            
            if session_id not in keep_ids:
                expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        app.logger.info(f"Removing session: {session_id}")
        current_sessions.pop(session_id, None)
        
    # Force garbage collection after cleanup
    gc.collect()
    app.logger.info(f"Cleanup finished. Removed {len(expired_sessions)} sessions. {len(current_sessions)} active sessions remain.")

def allowed_file(file):
    """Check if uploaded file has an allowed extension and MIME type"""
    if not file or not file.filename:
        return False
        
    has_valid_extension = '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    has_valid_mime = file.content_type in ALLOWED_MIME_TYPES
    return has_valid_extension and has_valid_mime

@app.route('/api/detect_sets', methods=['POST', 'OPTIONS'])
def detect_sets():
    """Handles image uploads and detects SETs."""
    # Handle OPTIONS requests for CORS preflight
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    # Check if models were loaded successfully
    if not models_available:
        return jsonify({'error': 'Model initialization failed. Please contact the administrator.'}), 503
    
    # Cleanup old sessions
    cleanup_old_sessions()
    
    # Limit active sessions to prevent memory overload
    if len(current_sessions) >= MAX_SESSIONS or check_memory_pressure():
        # Force cleanup to see if we can free up space
        cleanup_old_sessions(force=True)
        
        # If still over limit after cleanup, return busy error
        if len(current_sessions) >= MAX_SESSIONS or check_memory_pressure():
            return jsonify({'error': 'Server is currently busy. Please try again in a few minutes.'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file):
        return jsonify({'error': 'Invalid file type. Only PNG and JPEG are supported.'}), 400
    
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
        
        # For Railway deployment, construct URLs properly
        api_base = request.host_url.rstrip('/')
        
        response = jsonify({
            'session_id': session_id,
            'original_image_url': f"{api_base}/api/images/{session_id}/original",
            'processed_image_url': f"{api_base}/api/images/{session_id}/result",
            'detected_sets': detected_sets
        })
        
        # Ensure CORS headers are included
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error: ' + str(e)}), 500

@app.route('/api/images/<session_id>/<image_type>')
def get_session_image(session_id, image_type):
    """Serves images from in-memory session storage."""
    # Validate session exists
    if session_id not in current_sessions:
        app.logger.warning(f"Session {session_id} not found or expired")
        return jsonify({'error': 'Session not found or expired'}), 404
    
    # Validate image type
    if image_type not in ['original', 'result']:
        app.logger.warning(f"Invalid image type requested: {image_type}")
        return jsonify({'error': 'Invalid image type requested'}), 400
    
    session_data = current_sessions[session_id]
    
    # Get the appropriate image buffer
    if image_type == 'original':
        image_buffer = session_data.get('original')
    else:
        image_buffer = session_data.get('result')
    
    if image_buffer is None:
        app.logger.warning(f"Image {image_type} not found in session {session_id}")
        return jsonify({'error': 'Image not found in session'}), 404
    
    # Update session timestamp to extend TTL
    session_data['timestamp'] = time.time()
    
    # Create in-memory file-like object
    image_io = io.BytesIO(image_buffer)
    
    # Return the image with proper content type and CORS headers
    response = send_file(
        image_io,
        mimetype='image/jpeg',
        as_attachment=False,
        download_name=f"{session_id}_{image_type}.jpg"
    )
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/api/health')
def health_check():
    """Enhanced health check endpoint for monitoring."""
    try:
        # Get memory usage
        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_usage_mb = memory_usage / 1024  # Convert to MB on Linux
        
        # Models status - don't fail health check if models aren't loaded
        models_status = "available" if models_available else "initializing"
        
        # Calculate uptime
        uptime_seconds = time.time() - app_startup_time
        
        memory_info = {
            'active_sessions': len(current_sessions),
            'memory_usage_mb': round(memory_usage_mb, 2),
            'models_status': models_status,
            'uptime_seconds': round(uptime_seconds, 2),
            'worker_count': MAX_WORKERS,
            'max_sessions': MAX_SESSIONS,
            'session_ttl_seconds': SESSION_TTL
        }
        
        # Always report healthy for Railway health checks
        # This is crucial - we want the container to stay up even if models are still loading
        response = jsonify({
            'status': 'healthy',
            'models_status': models_status,
            'memory': memory_info
        })
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200
    except Exception as e:
        app.logger.error(f"Error in health check: {e}")
        # Still return status 200 for Railway health checks to pass
        response = jsonify({
            'status': 'warning',
            'error': str(e)
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200

# Railway expects root path to be accessible
@app.route('/')
def root():
    response = jsonify({
        'status': 'ok',
        'service': 'SET Detector API',
        'version': '1.0.0',
        'health_endpoint': '/api/health'
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# Debug endpoint to help troubleshoot model loading issues
@app.route('/api/debug/environment')
def debug_environment():
    """Endpoint for debugging environment variables and paths."""
    if not os.environ.get('DEBUG', 'False').lower() == 'true':
        return jsonify({'error': 'Debug endpoints disabled in production'}), 403
        
    try:
        env_vars = {key: value for key, value in os.environ.items() 
                   if 'API_KEY' not in key.upper() and 'SECRET' not in key.upper()}
        
        cwd = os.getcwd()
        directory_structure = {}
        
        # Check commonly used directories
        for path in [cwd, '/app', '/app/models']:
            if os.path.exists(path):
                try:
                    directory_structure[path] = os.listdir(path)
                except Exception as e:
                    directory_structure[path] = f"Error listing: {str(e)}"
            else:
                directory_structure[path] = "Path doesn't exist"
                
        models_info = "Not available - models not loaded"
        if models_available:
            try:
                # Try to get some basic model info without leaking sensitive details
                models_info = "Models loaded successfully"
            except:
                models_info = "Error retrieving model info"
                
        debug_info = {
            'environment': env_vars,
            'working_directory': cwd,
            'directory_structure': directory_structure,
            'models_info': models_info,
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        response = jsonify(debug_info)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        app.logger.error(f"Error in debug endpoint: {e}")
        return jsonify({'error': str(e)}), 500

# If this is the main module, run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
