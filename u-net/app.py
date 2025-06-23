from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import io
import base64
from test_model import ModelTester
import json

# Set matplotlib to use non-GUI backend before importing
import matplotlib
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no GUI)
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'api_uploads'
RESULTS_FOLDER = 'api_results'
MODEL_PATH = 'models/v9.pth'
CONFIG_PATH = 'configs/configV9.0.json'
DEFAULT_THRESHOLD = 0.98

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize model tester (load model once at startup)
print("Loading U-Net model...")
try:
    model_tester = ModelTester(MODEL_PATH, CONFIG_PATH, headless=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model_tester = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_tester is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Upload and process an image for weed detection
    
    Expected: multipart/form-data with 'image' file
    Optional: 'threshold' parameter (default: 0.98)
    
    Returns: JSON with image_id and processing status
    """
    try:
        if model_tester is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get threshold parameter
        threshold = float(request.form.get('threshold', DEFAULT_THRESHOLD))
        
        # Generate unique ID for this processing job
        image_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processing_time = datetime.now()
        
        # Save uploaded image
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.jpg', '.jpeg', '.png', '.bmp']:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        input_filename = f"{timestamp}_{image_id}_input{file_extension}"
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        file.save(input_path)
        
        # Process image with U-Net model
        print(f"Processing image: {input_filename}")
        
        # Generate output paths
        prediction_path = os.path.join(RESULTS_FOLDER, f"{timestamp}_{image_id}_prediction.png")
        overlay_borders_path = os.path.join(RESULTS_FOLDER, f"{timestamp}_{image_id}_prediction_overlay_borders.png")
        
        # Make prediction (single call to avoid duplicate processing)
        prediction, probabilities, resized_image, original_image = model_tester.predict(input_path, threshold)
        
        # Create mask borders and overlay
        mask_borders = model_tester.create_mask_borders(prediction)
        overlay_image = model_tester.overlay_mask_borders(resized_image, mask_borders)
        
        # Create visualization with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(resized_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Confidence map
        im1 = axes[0, 1].imshow(probabilities, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('Confidence Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Binary prediction
        axes[1, 0].imshow(prediction, cmap='gray')
        axes[1, 0].set_title(f'Prediction (threshold={threshold})')
        axes[1, 0].axis('off')
        
        # Original with mask borders
        axes[1, 1].imshow(overlay_image)
        axes[1, 1].set_title('Original + Mask Borders')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(prediction_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {prediction_path}")
        
        # Save the overlay borders image separately
        cv2.imwrite(overlay_borders_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
        print(f"Border overlay saved to: {overlay_borders_path}")
        
        # Close the figure to free memory
        plt.close()
        
        # Get statistics
        total_pixels = prediction.shape[0] * prediction.shape[1]
        weed_pixels = np.sum(prediction > threshold)
        weed_percentage = (weed_pixels / total_pixels) * 100
        
        print(f"Processing completed for {image_id}")
        
        return jsonify({
            'image_id': image_id,
            'timestamp': processing_time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'completed',
            'threshold': threshold,
            'statistics': {
                'weed_coverage_percent': round(weed_percentage, 2),
                'weed_pixels': int(weed_pixels),
                'total_pixels': int(total_pixels)
            },
            'files': {
                'input': input_filename,
                'prediction': f"{timestamp}_{image_id}_prediction.png",
                'overlay_borders': f"{timestamp}_{image_id}_prediction_overlay_borders.png"
            }
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/result/<image_id>/<result_type>', methods=['GET'])
def get_result(image_id, result_type):
    """
    Get processed result image
    
    Args:
        image_id: Unique identifier for the processing job
        result_type: 'prediction', 'overlay_borders', or 'input'
    
    Returns: Image file
    """
    try:
        # Find files with the image_id
        if result_type == 'input':
            folder = UPLOAD_FOLDER
            pattern = f"*_{image_id}_input.*"
        elif result_type == 'overlay_borders':
            folder = RESULTS_FOLDER
            pattern = f"*_{image_id}_prediction_overlay_borders.png"
        else:  # prediction
            folder = RESULTS_FOLDER
            pattern = f"*_{image_id}_prediction.png"
        
        # Find matching files
        import glob
        matching_files = glob.glob(os.path.join(folder, pattern))
        
        if not matching_files:
            return jsonify({'error': 'Result not found'}), 404
        
        file_path = matching_files[0]  # Take the first match
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, mimetype='image/png')
        
    except Exception as e:
        print(f"Error retrieving result: {e}")
        return jsonify({'error': f'Failed to retrieve result: {str(e)}'}), 500

@app.route('/results/<image_id>', methods=['GET'])
def get_results_info(image_id):
    """
    Get information about processed results
    
    Args:
        image_id: Unique identifier for the processing job
    
    Returns: JSON with available results and metadata
    """
    try:
        import glob
        
        # Find all files for this image_id
        input_files = glob.glob(os.path.join(UPLOAD_FOLDER, f"*_{image_id}_input.*"))
        result_files = glob.glob(os.path.join(RESULTS_FOLDER, f"*_{image_id}_*.png"))
        
        if not input_files and not result_files:
            return jsonify({'error': 'No results found for this image_id'}), 404
        
        available_results = []
        if input_files:
            available_results.append('input')
        
        for file_path in result_files:
            filename = os.path.basename(file_path)
            if 'prediction_overlay_borders' in filename:
                available_results.append('overlay_borders')
            elif 'prediction' in filename:
                available_results.append('prediction')
        
        return jsonify({
            'image_id': image_id,
            'available_results': available_results,
            'total_files': len(input_files) + len(result_files)
        })
        
    except Exception as e:
        print(f"Error getting results info: {e}")
        return jsonify({'error': f'Failed to get results info: {str(e)}'}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_old_files():
    """
    Clean up old files (optional maintenance endpoint)
    
    Returns: JSON with cleanup statistics
    """
    try:
        import time
        
        # Remove files older than 1 hour (3600 seconds)
        cutoff_time = time.time() - 3600
        
        cleaned_count = 0
        
        # Clean upload folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.getctime(file_path) < cutoff_time:
                os.remove(file_path)
                cleaned_count += 1
        
        # Clean results folder
        for filename in os.listdir(RESULTS_FOLDER):
            file_path = os.path.join(RESULTS_FOLDER, filename)
            if os.path.getctime(file_path) < cutoff_time:
                os.remove(file_path)
                cleaned_count += 1
        
        return jsonify({
            'status': 'completed',
            'files_cleaned': cleaned_count
        })
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Weed Detection API...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Config path: {CONFIG_PATH}")
    print(f"Default threshold: {DEFAULT_THRESHOLD}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print("\nAPI Endpoints:")
    print("- GET  /health                     - Health check")
    print("- POST /upload                     - Upload and process image")
    print("- GET  /result/<id>/<type>         - Get result image")
    print("- GET  /results/<id>               - Get results info")
    print("- POST /cleanup                    - Clean old files")
    print("\nStarting server on http://localhost:5001")
    
    app.run(debug=True, host='0.0.0.0', port=5001) 