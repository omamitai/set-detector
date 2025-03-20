#!/usr/bin/env python3
"""
Pre-launch validation script for SET detector application.
This script verifies key application components before the main app starts.
"""

import os
import sys
import logging
import pathlib
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("prelaunch")

def check_environment():
    """Check environment variables"""
    logger.info("Checking environment variables...")
    
    env_vars = [
        'MAX_WORKERS', 
        'MAX_SESSIONS', 
        'SESSION_TTL', 
        'CLEANUP_INTERVAL', 
        'MAX_MEMORY_PERCENT',
        'MODELS_DIR'
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"  ✓ {var}={value}")
        else:
            logger.warning(f"  ✗ {var} not set")

def check_filesystem():
    """Check file system for required files and directories"""
    logger.info("Checking filesystem...")
    
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    
    # Check for required files
    required_files = ['app.py', 'set_detector.py', 'requirements.txt']
    for file in required_files:
        if os.path.isfile(file):
            logger.info(f"  ✓ {file} exists")
        else:
            logger.error(f"  ✗ {file} missing")
    
    # Check for models directory
    models_dir = os.environ.get('MODELS_DIR', './models')
    logger.info(f"Models directory set to: {models_dir}")
    
    if os.path.isdir(models_dir):
        logger.info(f"  ✓ Models directory exists")
        
        # Check for model subdirectories
        subdirs = ['Card/16042024', 'Characteristics/11022025', 'Shape/15052024']
        for subdir in subdirs:
            full_path = os.path.join(models_dir, subdir)
            if os.path.isdir(full_path):
                logger.info(f"  ✓ {subdir} directory exists")
                
                # List files in this directory
                files = os.listdir(full_path)
                logger.info(f"    Files in {subdir}: {files}")
            else:
                logger.error(f"  ✗ {subdir} directory missing")
    else:
        logger.error(f"  ✗ Models directory missing")
        
        # Try to find models elsewhere
        logger.info("Searching for model files...")
        model_extensions = ['.pt', '.keras', '.yaml']
        for root, dirs, files in os.walk(cwd, topdown=True, followlinks=False):
            for file in files:
                if any(file.endswith(ext) for ext in model_extensions):
                    logger.info(f"  Found model file: {os.path.join(root, file)}")

def validate_imports():
    """Validate that all required imports are available"""
    logger.info("Validating imports...")
    
    required_modules = [
        'flask', 
        'numpy', 
        'tensorflow', 
        'torch', 
        'ultralytics', 
        'cv2', 
        'pandas'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"  ✓ {module} imported successfully")
        except ImportError as e:
            logger.error(f"  ✗ {module} import failed: {e}")

def main():
    """Run all validation checks"""
    logger.info("Starting pre-launch validation")
    
    start_time = time.time()
    
    try:
        check_environment()
        check_filesystem()
        validate_imports()
        
        logger.info(f"Validation completed in {time.time() - start_time:.2f} seconds")
        logger.info("Application appears ready for launch")
        return 0
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
