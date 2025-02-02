import os
import torch
import cv2
import numpy as np
import json
import joblib  # Use joblib for saving/loading models
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch.nn as nn
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define preprocessing for image input (same as training)
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Initialize FaceNet model for face embeddings
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load Classifier Model
class FaceClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(FaceClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Load the trained classifier model
classifier_model = FaceClassifier(embedding_dim=512, num_classes=3).to(device)
classifier_model.load_state_dict(torch.load('classifier_model.pth'))
classifier_model.eval()

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Load the pre-trained SVM model using joblib
# Check if the SVM model exists, and load it
if os.path.exists('svm_model.pkl'):
    svm_model = joblib.load('svm_model.pkl')
else:
    svm_model = SVC(kernel='linear', probability=True)  # Initialize if not found

# Training function
def train_model():
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = ImageFolder('/content/Dataset/Dataset/Dataset_cropped/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Extract embeddings for training
    def get_embeddings(data_loader):
        embeddings = []
        labels = []
        with torch.no_grad():
            for imgs, lbls in data_loader:
                imgs = imgs.to(device)
                emb = facenet_model(imgs).cpu().numpy()
                embeddings.append(emb)
                labels.append(lbls.numpy())
        return np.vstack(embeddings), np.hstack(labels)

    # Extract embeddings for train
    train_embeddings, train_labels = get_embeddings(train_loader)

    # Train SVM on the FaceNet embeddings
    svm_model.fit(train_embeddings, train_labels)
    joblib.dump(svm_model, 'svm_model.pkl')  # Save the trained SVM model

# Face recognition and bounding box for camera feed
def recognize_from_camera():
    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow to avoid MSMF issues on Windows

    # Define the confidence threshold for classification
    confidence_threshold = 0.8  # You can adjust this based on your preference

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Detect faces in the frame
        boxes, probs = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                # Draw bounding box on detected face
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            # Extract face embeddings from the frame
            faces = mtcnn(frame)
            if faces is not None:
                embeddings = facenet_model(faces).cpu().detach().numpy()
                predictions = classifier_model(torch.tensor(embeddings, dtype=torch.float32).to(device))

                # Get the max class probability and index
                probabilities = torch.nn.functional.softmax(predictions, dim=1)
                max_probs, predicted_classes = torch.max(probabilities, dim=1)

                # If the max probability is below the threshold, label it as 'unknown'
                for i, box in enumerate(boxes):
                    # Handle case for single detection
                    if len(predicted_classes) > i:  # Ensure we don't index out of range
                        class_name = "unknown" if max_probs[i].item() < confidence_threshold else class_names[predicted_classes[i].item()]
                    else:
                        class_name = "unknown"
                    
                    cv2.putText(frame, class_name,
                                (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == '__main__':
    # Check if models are trained, if not, train
    if not os.path.exists('classifier_model.pth'):
        print("Training models...")
        train_model()
        print("Model training completed!")

    # Start face recognition from camera feed
    print("Starting face recognition from camera feed...")
    recognize_from_camera()
