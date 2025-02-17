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
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from torchvision import transforms


app = Flask(__name__)

# Constants
FACE_SIMILARITY_THRESHOLD = 0.9
CONFIDENCE_THRESHOLD = 0.9
DETECTION_PROBABILITY_THRESHOLD = 0.95
DATASET_PATH = "Dataset/train"  # Base path for dataset storage
ATTENDANCE_LOG_PATH = "attendance_log.xlsx" # Update this path # Path to log Excel file
TRAIN_DIR = "Dataset/train"
TEST_DIR = "Dataset/test"
CLASS_NAMES_FILE = "class_names.json"

# Google API setup
SERVICE_ACCOUNT_FILE = 'admin-data.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)
sheets_service = build('sheets', 'v4', credentials=creds)

# Google Drive folder ID where the sheet should be stored
FOLDER_ID = "1KCkXb-o5kSKz4hqua1xldZ2pU0bjwdzq"

# Google Sheet ID (if exists, otherwise a new one will be created)
SPREADSHEET_ID = "1n-ZQq1UPlklsvERQl15ijsTueprSaIqNG29NOChnr3o"

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

# Function to load or update class names
def load_class_names():
    if os.path.exists(CLASS_NAMES_FILE):
        with open(CLASS_NAMES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_class_name(new_name):
    class_names = load_class_names()
    if new_name not in class_names:
        class_names.append(new_name)
        class_names = sorted(class_names, key=str.lower) 
        with open(CLASS_NAMES_FILE, 'w') as f:
            json.dump(class_names, f, indent=4)

# Capture and save images
@app.route('/start_capture', methods=['POST'])
def start_capture():
    try:
        data = request.get_json()
        user_name = data.get("name", "unknown")
        if not user_name:
            return jsonify({"error": "User name is required"}), 400

        user_train_dir = os.path.join(TRAIN_DIR, user_name)
        user_test_dir = os.path.join(TEST_DIR, user_name)
        os.makedirs(user_train_dir, exist_ok=True)
        os.makedirs(user_test_dir, exist_ok=True)

        save_class_name(user_name)  # Update class names file

        captured_images = []
        cap = cv2.VideoCapture(0)
        count = 0

        while count < 300:
            ret, frame = cap.read()
            if not ret:
                continue

            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        captured_images.append(face)
                        count += 1

            cv2.imshow("Face Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Split data into train (80%) and test (20%)
        train_split = int(len(captured_images) * 0.8)
        for i, img in enumerate(captured_images):
            img_path = os.path.join(user_train_dir if i < train_split else user_test_dir, f"{user_name}_{i+1}.jpg")
            cv2.imwrite(img_path, img)

        return jsonify({"message": "Capture completed successfully", "train_images": train_split, "test_images": len(captured_images) - train_split})

    except Exception as e:
        return jsonify({"error": str(e), "message": "Error capturing images"})

@app.route('/update_dataset', methods=['POST'])
def update_dataset():
    """Receive captured images, store them in train/test folders, and update class names."""
    try:
        data = request.get_json()
        user_name = data.get("name")
        images = data.get("images", [])

        if not user_name or not images:
            return jsonify({"error": "Invalid data"}), 400

        # Create user directories
        user_train_dir = os.path.join(TRAIN_DIR, user_name)
        user_test_dir = os.path.join(TEST_DIR, user_name)
        os.makedirs(user_train_dir, exist_ok=True)
        os.makedirs(user_test_dir, exist_ok=True)

        # Split images (80% for train, 20% for test)
        total_images = len(images)
        train_count = int(0.8 * total_images)
        test_count = total_images - train_count

        for i, img_data in enumerate(images):
            img_bytes = base64.b64decode(img_data.split(",")[1])
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Save image
            file_name = f"{user_name}_{i+1}.jpg"
            folder = user_train_dir if i < train_count else user_test_dir
            cv2.imwrite(os.path.join(folder, file_name), img)

        # Update class names
        save_class_name(user_name)

        return jsonify({
            "message": "Images processed successfully",
            "processed_faces": total_images,
            "train_count": train_count,
            "test_count": test_count,
            "failed_count": 0
        })

    except Exception as e:
        return jsonify({"error": str(e), "message": "Error processing images"}), 500

# Load class names
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

# Function to check if the Google Sheet exists
def check_google_sheet_exists(spreadsheet_id):
    try:
        sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        return True
    except Exception as e:
        print(f"Google Sheet check failed: {e}")
        return False

# Function to create a new Google Sheet inside the specified folder
def create_google_sheet(sheet_name):
    try:
        spreadsheet = {
            "properties": {"title": sheet_name},
            "sheets": [{"properties": {"title": "Sheet1"}}]
        }
        sheet = sheets_service.spreadsheets().create(body=spreadsheet).execute()
        spreadsheet_id = sheet["spreadsheetId"]

        # Move the sheet to the specified folder in Google Drive
        drive_service.files().update(
            fileId=spreadsheet_id,
            addParents=FOLDER_ID,
            removeParents="root",
            fields="id, parents"
        ).execute()

        print(f"Google Sheet created and moved to Drive folder: {spreadsheet_id}")

        # Add headers after sheet creation
        add_sheet_headers(spreadsheet_id)

        return spreadsheet_id
    except Exception as e:
        print(f"Error creating Google Sheet: {e}")
        return None

# Function to add headers to the Google Sheet
def add_sheet_headers(spreadsheet_id):
    try:
        values = [["Date", "Name", "Enter Time", "Exit Time"]]
        body = {"values": values}
        sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range="Sheet1!A1:D1",
            valueInputOption="RAW",
            body=body
        ).execute()
        print("Headers added successfully.")
    except Exception as e:
        print(f"Error adding headers: {e}")

# Function to log attendance with Enter/Exit times
def log_attendance(spreadsheet_id, name, action):
    date_today = time.strftime("%Y-%m-%d")
    current_time = time.strftime("%H:%M:%S")

    try:
        sheet_data = sheets_service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range="Sheet1!A:D"
        ).execute()
        values = sheet_data.get("values", [])

        # If sheet is empty, add headers
        if not values:
            add_sheet_headers(spreadsheet_id)
            values.append(["Date", "Name", "Enter Time", "Exit Time"])

        updated = False
        for row in values[1:]:  # Skipping header row
            if len(row) < 4:  
                row.extend([""] * (4 - len(row)))  # Ensure 4 columns exist
            
            if row[0] == date_today and row[1] == name:
                if action == "enter":
                    return {"message": "Entry already recorded."}
                elif action == "exit":
                    row[3] = current_time  # Update exit time
                    updated = True
                    break

        if not updated:
            if action == "enter":
                values.append([date_today, name, current_time, ""])
            elif action == "exit":
                return {"error": "Cannot exit without entering first."}

        # Update Google Sheets
        sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range="Sheet1!A:D",
            valueInputOption="RAW",
            body={"values": values}
        ).execute()

        return {"message": "Attendance logged successfully."}

    except Exception as e:
        print(f"Error logging attendance: {e}")
        return {"error": "Failed to log attendance."}

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/update')
def update_page():
    return render_template('update.html')  # Ensure this file exists in the templates folder

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
                log_attendance(SPREADSHEET_ID, class_name, action)  # Use the action parameter from the request

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