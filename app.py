from flask import Flask, render_template, request, jsonify,send_from_directory
import torch
import json
import os 
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageFile
import base64
from io import BytesIO
import logging
import shutil
import cv2
import time
from datetime import datetime

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = 'models'
DATASET_DIR = 'Dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'Train')
TEST_DIR = os.path.join(DATASET_DIR, 'Test')
CLASS_NAMES_FILE = os.path.join(MODEL_DIR, 'class_names.json')
BACKUP_DIR = 'backups'
MAX_IMAGE_SIZE = (640, 480)  # Maximum image dimensions
FACE_SIZE = (160, 160)  # Standard size for face images

# Create necessary directories
for directory in [MODEL_DIR, DATASET_DIR, TRAIN_DIR, TEST_DIR, BACKUP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize models
try:
    mtcnn = MTCNN(
        image_size=160, 
        margin=10, 
        keep_all=True,
        min_face_size=20,
        factor=0.709,
        post_process=True,
        device=device
    )
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    logger.info("Models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing models: {e}")
    raise

# Load or create class names
def load_class_names():
    try:
        if os.path.exists(CLASS_NAMES_FILE):
            with open(CLASS_NAMES_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading class names: {e}")
        return {}

class_names = load_class_names()

def backup_dataset():
    """Create a backup of the dataset"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(BACKUP_DIR, f'dataset_backup_{timestamp}')
        shutil.copytree(DATASET_DIR, backup_path)
        logger.info(f"Dataset backed up to {backup_path}")
    except Exception as e:
        logger.error(f"Backup failed: {e}")

def process_image(image):
    """Process and validate image"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
            image.thumbnail(MAX_IMAGE_SIZE)
        
        return image
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def detect_and_align_face(image):
    """Detect and align face in image"""
    try:
        # Detect face
        boxes, probs = mtcnn.detect(image)
        
        if boxes is None:
            return None
        
        # Get the face with highest probability
        if len(probs) > 0:
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            
            # Extract and align face
            face = mtcnn(image)
            if face is not None:
                return face
                
        return None
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return None

def save_face_image(face_tensor, directory, name, index):
    """Save face tensor as image"""
    try:
        if face_tensor is None:
            return False
            
        # Convert tensor to image
        face_image = face_tensor.squeeze(0).permute(1, 2, 0).numpy()
        face_image = (face_image * 255).astype(np.uint8)
        face_pil = Image.fromarray(face_image)
        
        # Ensure face is the right size
        if face_pil.size != FACE_SIZE:
            face_pil = face_pil.resize(FACE_SIZE)
            
        # Save image
        filename = f"{name}_{index:03d}.jpg"
        filepath = os.path.join(directory, filename)
        face_pil.save(filepath, quality=95)
        return True
    except Exception as e:
        logger.error(f"Error saving face image: {e}")
        return False

def process_and_save_faces(image_data, name):
    """Process multiple images and save detected faces"""
    try:
        # Create user directories
        user_train_dir = os.path.join(TRAIN_DIR, name)
        user_test_dir = os.path.join(TEST_DIR, name)
        os.makedirs(user_train_dir, exist_ok=True)
        os.makedirs(user_test_dir, exist_ok=True)

        total_images = len(image_data)
        train_split = int(total_images * 0.8)  # 80% for training
        processed_faces = 0
        train_count = 0
        test_count = 0
        failed_count = 0

        for i, base64_image in enumerate(image_data):
            try:
                # Convert base64 to image
                image_data = base64_image.split(',')[1]
                image = Image.open(BytesIO(base64.b64decode(image_data)))
                
                # Process image
                processed_image = process_image(image)
                if processed_image is None:
                    failed_count += 1
                    continue

                # Detect and align face
                face_tensor = detect_and_align_face(processed_image)
                if face_tensor is None:
                    failed_count += 1
                    continue

                # Save face
                if i < train_split:
                    if save_face_image(face_tensor, user_train_dir, name, train_count):
                        train_count += 1
                        processed_faces += 1
                else:
                    if save_face_image(face_tensor, user_test_dir, name, test_count):
                        test_count += 1
                        processed_faces += 1

            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                failed_count += 1
                continue

        return {
            "success": True,
            "processed_faces": processed_faces,
            "train_count": train_count,
            "test_count": test_count,
            "failed_count": failed_count
        }

    except Exception as e:
        logger.error(f"Error in process_and_save_faces: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('update_page.html')

@app.route('/update_dataset', methods=['POST'])
def update_dataset():
    """Handle dataset updates"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        if 'images' not in data or 'name' not in data:
            return jsonify({"error": "Images and name are required"}), 400

        name = data['name'].strip()
        if not name:
            return jsonify({"error": "Invalid name"}), 400

        # Create backup before updating
        backup_dataset()

        # Update class names
        if name not in class_names.values():
            new_class_id = str(len(class_names))
            class_names[new_class_id] = name
            with open(CLASS_NAMES_FILE, 'w') as f:
                json.dump(class_names, f, indent=4)

        # Process images
        result = process_and_save_faces(data['images'], name)
        
        if not result["success"]:
            return jsonify({"error": result["error"]}), 500

        return jsonify({
            "message": "Dataset updated successfully",
            "processed_faces": result["processed_faces"],
            "train_count": result["train_count"],
            "test_count": result["test_count"],
            "failed_count": result["failed_count"]
        })

    except Exception as e:
        logger.error(f"Error in update_dataset: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_users', methods=['GET'])
def get_users():
    """Get list of registered users"""
    try:
        users = list(class_names.values())
        return jsonify({
            "users": users,
            "count": len(users)
        })
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/delete_user', methods=['POST'])
def delete_user():
    """Delete a user and their data"""
    try:
        data = request.json
        if 'name' not in data:
            return jsonify({"error": "Name is required"}), 400

        name = data['name'].strip()
        
        # Create backup before deletion
        backup_dataset()

        # Remove from class names
        class_names_updated = {k: v for k, v in class_names.items() if v != name}
        with open(CLASS_NAMES_FILE, 'w') as f:
            json.dump(class_names_updated, f, indent=4)

        # Remove directories
        train_dir = os.path.join(TRAIN_DIR, name)
        test_dir = os.path.join(TEST_DIR, name)
        
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        return jsonify({"message": f"User {name} deleted successfully"})

    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({
        "error": "An unexpected error occurred",
        "details": str(error)
    }), 500



if __name__ == '__main__':
    logger.info("Starting Face Recognition Training Server")
    app.run(debug=True, host='0.0.0.0', port=5000)