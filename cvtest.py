import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Import YOLO
from yolov5 import detect as yolo_detect

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))
        
class ResNet(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_object_from_frame(frame, yolo_model):
    """
    Detect objects in frame using YOLOv5 and return the largest detected object
    """
    # Run YOLOv5 detection
    results = yolo_model(frame)
    
    # Get detection results
    detections = results.xyxy[0].cpu().numpy()  # xyxy format: x1, y1, x2, y2, confidence, class
    
    if len(detections) == 0:
        return None, None
    
    # Get the detection with highest confidence
    best_detection = detections[np.argmax(detections[:, 4])]
    x1, y1, x2, y2 = map(int, best_detection[:4])
    
    # Crop the detected object
    object_frame = frame[y1:y2, x1:x2]
    
    if object_frame.size == 0:
        return None, None
        
    return object_frame, (x1, y1, x2, y2)

def predict_frame(frame, classification_model, device, transformations, classes):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    image = Image.fromarray(frame_rgb)
    
    # Apply transformations
    image_tensor = transformations(image)
    
    # Add batch dimension and move to device
    image_tensor = to_device(image_tensor.unsqueeze(0), device)
    
    # Get predictions
    with torch.no_grad():
        outputs = classification_model(image_tensor)
        prob, preds = torch.max(outputs, dim=1)
        predicted_class = classes[preds[0].item()]
        confidence = prob[0].item()
    
    return predicted_class, confidence

def main():
    # Initialize device and models
    device = get_default_device()
    
    # Load YOLOv5 model
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    yolo_model.to(device)
    
    # Define transformations for classification
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Define classes for classification (update this to match your trained model)
    classes = ['glass', 'metal', 'plastic']  # Updated to 8 classes
    
    # Initialize classification model with correct number of classes
    classification_model = ResNet(len(classes))
    
    # Load the trained weights
    model_path = "garbage_classifier.pth"
    if Path(model_path).exists():
        classification_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Error: Could not find model weights at {model_path}")
        return
    
    classification_model = to_device(classification_model, device)
    classification_model.eval()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect object in frame
        object_frame, bbox = get_object_from_frame(frame, yolo_model)
        
        if object_frame is not None:
            # Make classification prediction on detected object
            predicted_class, confidence = predict_frame(object_frame, classification_model, device, transformations, classes)
            
            # Draw bounding box and prediction on frame
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{predicted_class} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Object Detection and Classification', frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()