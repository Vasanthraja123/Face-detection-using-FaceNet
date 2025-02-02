import base64
import json
from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Home.html')

# Initialize MTCNN for face detection and FaceNet for recognition
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load Classifier Model
class FaceClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(FaceClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

classifier_model = FaceClassifier(embedding_dim=512, num_classes=3).to(device)
classifier_model.load_state_dict(torch.load('classifier_model.pth'))
classifier_model.eval()

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        data = request.get_json()
        image_data = data['image']
        action = data['action']

        # Convert image from base64 to image array
        img_data = image_data.split(",")[1]
        img_bytes = BytesIO(base64.b64decode(img_data))
        img = Image.open(img_bytes)
        img = np.array(img)

        # Detect faces
        boxes, _ = mtcnn.detect(img)

        if boxes is None:
            return jsonify({"message": "No face detected"})

        faces = mtcnn(img)

        if faces is None:
            return jsonify({"message": "No faces detected in image"})

        embeddings = facenet_model(faces).cpu().detach().numpy()
        predictions = classifier_model(torch.tensor(embeddings, dtype=torch.float32).to(device))

        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        max_probs, predicted_classes = torch.max(probabilities, dim=1)

        recognized_names = []
        for i, prob in enumerate(max_probs):
            class_name = class_names[predicted_classes[i].item()] if prob > 0.8 else "unknown"
            recognized_names.append({"name": class_name, "confidence": prob.item()})

        # Draw bounding boxes on the image
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            # Draw rectangle on the image (bounding box)
            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            # Add label with name and confidence score
            label = f"{recognized_names[i]['name']} ({recognized_names[i]['confidence']*100:.2f}%)"
            cv2.putText(img, label, (int(x_min), int(y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the image with bounding boxes locally
        output_image_path = 'V:\FaceNet 2\Output images/output_image.jpg'
        cv2.imwrite(output_image_path, img)

        # Convert the modified image to base64 to return to the frontend
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare the response data with the image and recognized names
        response_data = {
            "results": recognized_names,
            "image": f"data:image/jpeg;base64,{img_base64}",
            "output_image_path": output_image_path  # Path to the saved image
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
