import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import cv2
import json

def train_recognition_model():
    try:
        # Initialize device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained FaceNet model
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        # Preprocessing transformation for FaceNet
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Load Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        def crop_face(image_path):
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Take the first detected face
                face = image[y:y+h, x:x+w]
                return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            else:
                return Image.open(image_path)
        
        
        def get_embeddings(data_loader, model):
            embeddings = []
            labels = []
            
            with torch.no_grad():
                for imgs, lbls in data_loader:
                    imgs = imgs.to(device)
                    emb = model(imgs).cpu().numpy()
                    embeddings.append(emb)
                    labels.append(lbls.numpy())
            
            return np.vstack(embeddings), np.hstack(labels)
        
        # Load training and test datasets
        train_dataset = datasets.ImageFolder('Dataset//train', transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        test_dataset = datasets.ImageFolder('Dataset//test', transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Get class names
        class_names = train_dataset.classes
        print(f"Classes: {class_names}")
        
        # Extract embeddings
        print("Extracting embeddings...")
        train_embeddings, train_labels = get_embeddings(train_loader, facenet_model)
        test_embeddings, test_labels = get_embeddings(test_loader, facenet_model)
        
        print("Train embeddings shape:", train_embeddings.shape)
        print("Test embeddings shape:", test_embeddings.shape)
        
        np.save('train_embeddings.npy', train_embeddings)
        np.save('train_labels.npy', train_labels)
        np.save('test_embeddings.npy', test_embeddings)
        np.save('test_labels.npy', test_labels)
        
        # Train SVM classifier
        print("Training SVM classifier...")
        svm_model = SVC(kernel='linear', probability=True)
        svm_model.fit(train_embeddings, train_labels)
        joblib.dump(svm_model, 'svm_model.pkl')
        
        # Evaluate SVM model
        predictions = svm_model.predict(test_embeddings)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"SVM Test Accuracy: {accuracy * 100:.2f}%")
        
        # Define and train neural network classifier
        class FaceClassifier(nn.Module):
            def __init__(self, embedding_dim, num_classes):
                super(FaceClassifier, self).__init__()
                self.fc = nn.Linear(embedding_dim, num_classes)
            
            def forward(self, x):
                return self.fc(x)
        
        # Train neural network classifier
        print("Training neural network classifier...")
        num_classes = len(class_names)
        classifier_model = FaceClassifier(embedding_dim=512, num_classes=num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)

        for epoch in range(10):
            classifier_model.train()
            for embeddings, labels in zip(train_embeddings, train_labels):
                embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)

                outputs = classifier_model(embeddings)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print("Training completed!")
        
        torch.save(facenet_model.state_dict(), 'facenet_model.pth')
        torch.save(classifier_model.state_dict(), 'classifier_model.pth')
        
        with open('class_names.json', 'w') as f:
            json.dump(class_names, f)
        
        print("Training completed successfully!")
        return True, "Training completed successfully"
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False, str(e)

if __name__ == "__main__":
    success, message = train_recognition_model()
    if success:
        print("Training completed successfully")
    else:
        print(f"Training failed: {message}")