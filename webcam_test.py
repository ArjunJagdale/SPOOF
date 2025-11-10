import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import time

# ========================================
# SpoofNet Model Definition (same as training)
# ========================================
class SpoofNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(SpoofNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

# ========================================
# MediaPipe Face Cropping (SAME AS TRAINING)
# ========================================
def mediapipe_crop_coords(landmarks, img_w, img_h,
                          top_margin=0.15, bottom_margin=0.05,
                          side_margin=0.1):
    """
    Tight face-only crop for spoof detection:
    - Ear to ear horizontally
    - Hairline to chin vertically
    - No background → eliminates environmental bias
    - Forces model to learn facial texture artifacts only
    """
    lm_top = landmarks[10]
    lm_chin = landmarks[152]
    lm_left = landmarks[234]
    lm_right = landmarks[454]

    x_top = int(lm_top.x * img_w)
    y_top = int(lm_top.y * img_h)
    x_chin = int(lm_chin.x * img_w)
    y_chin = int(lm_chin.y * img_h)
    left_x = int(lm_left.x * img_w)
    right_x = int(lm_right.x * img_w)

    face_w = right_x - left_x
    face_h = y_chin - y_top

    x1 = max(0, int(left_x - face_w * side_margin))
    x2 = min(img_w, int(right_x + face_w * side_margin))
    y1 = max(0, int(y_top - face_h * top_margin))
    y2 = min(img_h, int(y_chin + face_h * bottom_margin))

    return x1, y1, x2, y2

# ========================================
# Blur Detection using Laplacian Variance
# ========================================
def calculate_blur_score(image):
    """
    Calculate blur score using Laplacian variance method.
    Higher score = sharper image
    Lower score = blurrier image
    
    Returns:
        blur_score: float - variance of Laplacian
        is_clear: bool - True if image is clear enough
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()
    
    # Threshold for clear image (adjust based on testing)
    # Typical values: < 100 = very blurry, 100-300 = moderate, > 300 = sharp
    BLUR_THRESHOLD = 150  # Adjust this value if needed
    is_clear = blur_score > BLUR_THRESHOLD
    
    return blur_score, is_clear

# ========================================
# Load Model
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SpoofNet(num_classes=2, dropout_rate=0.5)
model.load_state_dict(torch.load('best_spoofnet_model.pth', map_location=device))
model.to(device)
model.eval()

print("✓ Model loaded successfully!")

# ========================================
# Preprocessing Transform (SAME AS TRAINING)
# ========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========================================
# Initialize MediaPipe Face Mesh
# ========================================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========================================
# Webcam Inference with Face Cropping and Blur Detection
# ========================================
cam_source = 1  # Change to 0 or 2 if needed

cap = cv2.VideoCapture(cam_source)

if not cap.isOpened():
    print(f"Error: Could not open camera {cam_source}")
    print("Try changing cam_source to 0, 1, or 2")
    exit()

print("\n" + "="*60)
print("SPOOFNET - REAL-TIME FACE LIVENESS DETECTION")
print("="*60)
print("🎥 Using MediaPipe face cropping (same as training data)")
print("📌 Green box = Face detected and cropped")
print("🔵 Blue box = Face too blurry (move closer or improve lighting)")
print("🔴 Red prediction = FAKE/SPOOF detected")
print("🟢 Green prediction = REAL/LIVE face detected")
print("\nPress 'q' to quit")
print("="*60 + "\n")

# For FPS calculation
prev_time = time.time()
frame_count = 0
fps_display = 0

# Smoothing predictions
prediction_history = []
history_size = 5

# Blur score history for smoothing
blur_history = []
blur_history_size = 3

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Rotate if needed (uncomment if using rotated camera like in your capture script)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    img_h, img_w = frame.shape[:2]
    display = frame.copy()
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    prediction_text = "No face detected"
    confidence_text = ""
    box_color = (128, 128, 128)  # Gray for no detection
    blur_info = ""
    
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        x1, y1, x2, y2 = mediapipe_crop_coords(lm, img_w, img_h)
        
        # Check if crop is valid
        if (x2 - x1) > 60 and (y2 - y1) > 80:
            # Extract the cropped face
            crop = frame[y1:y2, x1:x2]
            
            # Calculate blur score
            blur_score, is_clear = calculate_blur_score(crop)
            
            # Smooth blur detection
            blur_history.append(is_clear)
            if len(blur_history) > blur_history_size:
                blur_history.pop(0)
            
            # Majority voting for blur
            is_face_clear = sum(blur_history) > len(blur_history) // 2
            
            blur_info = f"Blur: {blur_score:.1f}"
            
            if not is_face_clear:
                # Face is too blurry - show warning
                box_color = (255, 128, 0)  # Orange/Blue
                prediction_text = "⚠️ FACE TOO BLURRY"
                confidence_text = "Move closer or improve lighting"
                
                # Draw orange box
                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)
                
                # Clear prediction history when blurry
                prediction_history.clear()
                
            else:
                # Face is clear - proceed with inference
                # Draw the crop box in green
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Convert to PIL Image for transforms
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(crop_rgb)
                
                # Preprocess
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    pred_class = predicted.item()
                    conf_score = confidence.item() * 100
                
                # Smooth predictions using history
                prediction_history.append(pred_class)
                if len(prediction_history) > history_size:
                    prediction_history.pop(0)
                
                # Majority voting
                if len(prediction_history) >= 3:  # Wait for at least 3 predictions
                    smoothed_pred = max(set(prediction_history), key=prediction_history.count)
                    
                    # Determine label and color
                    # Class 0 = fake/spoof, Class 1 = real/live
                    if smoothed_pred == 1:
                        prediction_text = "✅ REAL / LIVE"
                        box_color = (0, 255, 0)  # Green
                    else:
                        prediction_text = "🚫 FAKE / SPOOF"
                        box_color = (0, 0, 255)  # Red
                    
                    confidence_text = f"Confidence: {conf_score:.1f}%"
                else:
                    prediction_text = "Analyzing..."
                    box_color = (255, 255, 0)  # Yellow
            
            # Draw cropped face in corner for visualization
            crop_small = cv2.resize(crop, (150, 150))
            display[10:160, img_w-160:img_w-10] = crop_small
            cv2.rectangle(display, (img_w-160, 10), (img_w-10, 160), box_color, 2)
    
    # Calculate FPS
    frame_count += 1
    curr_time = time.time()
    if curr_time - prev_time >= 1.0:
        fps_display = frame_count / (curr_time - prev_time)
        frame_count = 0
        prev_time = curr_time
    
    # Draw semi-transparent overlay for text
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (img_w, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
    
    # Draw prediction text
    cv2.putText(display, prediction_text, 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, box_color, 3)
    
    if confidence_text:
        cv2.putText(display, confidence_text, 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw blur score
    if blur_info:
        blur_color = (0, 255, 0) if is_face_clear else (0, 165, 255)
        cv2.putText(display, blur_info, 
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blur_color, 2)
    
    # Draw FPS
    cv2.putText(display, f"FPS: {fps_display:.1f}", 
                (img_w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw cropped face label
    if results.multi_face_landmarks:
        cv2.putText(display, "Cropped Face", 
                    (img_w - 155, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display frame
    cv2.imshow('SpoofNet - Face Liveness Detection', display)
    
    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()

print("\n" + "="*60)
print("✓ Webcam test completed")
print("="*60)