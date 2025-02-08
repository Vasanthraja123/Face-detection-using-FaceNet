import base64
import json
import os
import time
import pandas as pd
from flask import Flask, request, jsonify, render_template
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from markupsafe import Markup, escape

app = Flask(__name__)

# Constants
FACE_SIMILARITY_THRESHOLD = 0.9
CONFIDENCE_THRESHOLD = 0.9
DETECTION_PROBABILITY_THRESHOLD = 0.95
DATASET_PATH = "Dataset/train"  # Base path for dataset storage
ATTENDANCE_LOG_PATH = "attendance_log.xlsx"  # Path to log Excel file

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

class FaceClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(FaceClassifier, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)

# Load class names and dynamically set num_classes
def load_class_names():
    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        print(f"Error loading class names: {str(e)}")
        return []

class_names = load_class_names()

def save_cropped_face(image, box, username):
    """
    Crop and save face image to the appropriate dataset directory
    """
    try:
        # Create user directory if it doesn't exist
        user_dir = os.path.join(DATASET_PATH, username)
        os.makedirs(user_dir, exist_ok=True)
        
        # Convert box coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        
        # Crop face from image
        face_img = image[y1:y2, x1:x2]
        
        # Generate unique filename using timestamp
        filename = f"face_{int(time.time())}_{np.random.randint(1000)}.jpg"
        filepath = os.path.join(user_dir, filename)
        
        # Save cropped face
        cv2.imwrite(filepath, face_img)
        return filepath
    except Exception as e:
        print(f"Error saving cropped face: {str(e)}")
        return None

# Load models and reference embeddings
try:
    classifier_model = FaceClassifier(embedding_dim=512, num_classes=len(class_names)).to(device)
    classifier_model.load_state_dict(torch.load('classifier_model.pth'))
    classifier_model.eval()

    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    
    # Load reference embeddings for each known person
    reference_embeddings = {}
    reference_dir = 'reference_embeddings'
    if os.path.exists(reference_dir):
        for person in os.listdir(reference_dir):
            person_embeddings = []
            person_path = os.path.join(reference_dir, person)
            for embedding_file in os.listdir(person_path):
                if embedding_file.endswith('.npy'):
                    embedding = np.load(os.path.join(person_path, embedding_file))
                    person_embeddings.append(embedding)
            if person_embeddings:
                reference_embeddings[person] = np.stack(person_embeddings)

except Exception as e:
    print(f"Error loading models and references: {str(e)}")
    raise

def check_face_similarity(embedding, reference_embeddings):
    """
    Check if a face embedding is similar to any known reference embeddings
    Returns the most similar person and the similarity score
    """
    max_similarity = 0
    most_similar_person = None
    
    embedding = embedding.cpu().numpy()
    
    for person, ref_embeddings in reference_embeddings.items():
        similarities = cosine_similarity(embedding.reshape(1, -1), ref_embeddings)
        similarity = np.max(similarities)
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_person = person
    
    return most_similar_person, max_similarity

def log_attendance(name, action):
    """
    Log attendance (entry or exit) into an Excel file.
    """
    try:
        # Check if the attendance log file exists, if not create it
        if os.path.exists(ATTENDANCE_LOG_PATH):
            df = pd.read_excel(ATTENDANCE_LOG_PATH)
        else:
            df = pd.DataFrame(columns=["Name", "Action", "Date", "Time"])

        # Get the current date and time
        current_time = time.strftime('%H:%M:%S')
        current_date = time.strftime('%Y-%m-%d')

        # Append new log entry
        new_log = pd.DataFrame([[name, action, current_date, current_time]], columns=["Name", "Action", "Date", "Time"])
        df = pd.concat([df, new_log], ignore_index=True)

        # Save the log to the Excel file
        df.to_excel(ATTENDANCE_LOG_PATH, index=False)

    except Exception as e:
        print(f"Error logging attendance: {str(e)}")


@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/update')
def update():
    return render_template('update_page.html')
@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        data = request.get_json()
        image_data = data['image']
        action = data.get('action', 'enter')  # Get the action from request, default to 'enter'

        # Convert image
        img_data = image_data.split(",")[1]
        img_bytes = BytesIO(base64.b64decode(img_data))
        img = Image.open(img_bytes)
        img = np.array(img)

        # Detect faces
        boxes, probs = mtcnn.detect(img)
        
        if boxes is None or len(boxes) == 0:
            return jsonify({"message": "No faces detected"})

        # Filter faces based on detection probability
        valid_face_indices = [i for i, prob in enumerate(probs) if prob > DETECTION_PROBABILITY_THRESHOLD]
        if not valid_face_indices:
            return jsonify({"message": "No faces detected with high confidence"})

        boxes = boxes[valid_face_indices]
        faces = mtcnn(img)

        if faces is None or len(faces) == 0:
            return jsonify({"message": "Failed to extract face features"})

        # Get embeddings
        with torch.no_grad():
            embeddings = facenet_model(faces)
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        # Get classifier predictions
        predictions = classifier_model(embeddings)
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        max_probs, predicted_classes = torch.max(probabilities, dim=1)

        recognized_names = []
        saved_faces = []  # Track saved face locations
        
        for i, (embedding, prob) in enumerate(zip(embeddings, max_probs)):
            # First check similarity with reference embeddings
            similar_person, similarity_score = check_face_similarity(embedding, reference_embeddings)
            
            if similarity_score > FACE_SIMILARITY_THRESHOLD:
                # Use the person identified through similarity
                class_name = similar_person
                confidence = similarity_score
            elif prob > CONFIDENCE_THRESHOLD:
                # Use classifier prediction if confidence is high enough
                class_name = class_names[predicted_classes[i].item()]
                confidence = prob.item()
            else:
                # Mark as unknown if neither criterion is met
                class_name = "Unknown Person"
                confidence = 0.0

            # Save cropped face if person is known
            saved_path = None
            if class_name != "Unknown Person":
                saved_path = save_cropped_face(img, boxes[i], class_name)

            recognized_names.append({
                "name": class_name,
                "confidence": confidence,
                "box": boxes[i].tolist(),
                "saved_path": saved_path
            })

            # Log attendance based on recognized person
            if class_name != "Unknown Person":
                log_attendance(class_name, action)  # Use the action parameter from the request

        # Draw annotations
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for recognition in recognized_names:
            box = recognition["box"]
            name = recognition["name"]
            conf = recognition["confidence"]
            
            if name != "Unknown Person":
                color = (0, 255, 0)  # Green for known faces
                label = f"{name} ({conf*100:.1f}%)"
            else:
                color = (0, 0, 255)  # Red for unknown faces
                label = "Unknown Person"
            
            cv2.rectangle(img_rgb, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         color, 2)
            cv2.putText(img_rgb, label, 
                       (int(box[0]), int(box[1]-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save and return result
        output_path = 'Output images/output_image.jpg'
        cv2.imwrite(output_path, img_rgb)

        _, buffer = cv2.imencode('.jpg', img_rgb)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "results": recognized_names,
            "image": f"data:image/jpeg;base64,{img_base64}",
            "output_image_path": output_path,
            "saved_faces": [r["saved_path"] for r in recognized_names if r["saved_path"]]
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Error processing image"
        })

if __name__ == '__main__':
    app.run(debug=True)