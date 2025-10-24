import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import mediapipe as mp
import numpy as np
from collections import deque
from PIL import Image

# ========================
#  SECURITY CONFIGURATION
# ========================
REAL_CONFIDENCE_THRESHOLD = 0.80  # Require 75% confidence for "real"
FRAME_BUFFER_SIZE = 10  # Number of frames to consider
CONSISTENCY_THRESHOLD = 0.8  # 80% of frames must agree
ENABLE_MULTI_FRAME = True  # Toggle multi-frame verification

# Brightness check thresholds
BRIGHTNESS_TOO_LOW = 50  # Below this = too dark
BRIGHTNESS_TOO_HIGH = 220  # Above this = too bright (overexposed)
BRIGHTNESS_RATIO_THRESHOLD = 0.3  # 30% of pixels must be in valid range

# Face mesh completeness check
MIN_LANDMARKS_VISIBLE = 0.95  # 95% of landmarks must be visible
ENABLE_MESH_CHECK = True  # Toggle mesh completeness verification

# Sunglasses/Occlusion check
ENABLE_SUNGLASSES_CHECK = True
EYE_REGION_DARKNESS_THRESHOLD = 80  # Mean brightness below this suggests occlusion


# ========================
#  SpoofNet architecture
# ========================
class SpoofNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(SpoofNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.conv4(x); x = self.conv5(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x


# ========================
#  Mediapipe Crop Utils
# ========================
def mediapipe_crop_coords(landmarks, img_w, img_h, top_offset_ratio=0.03, lr_margin_ratio=0.12):
    lm_top = landmarks[10]
    lm_chin = landmarks[152]
    left_x = int(landmarks[234].x * img_w)
    right_x = int(landmarks[454].x * img_w)
    x_top = int(lm_top.x * img_w)
    y_top = int(lm_top.y * img_h)
    x_chin = int(lm_chin.x * img_w)
    y_chin = int(lm_chin.y * img_h)
    face_w = right_x - left_x
    pad_x = int(face_w * lr_margin_ratio)
    pad_top = int((y_chin - y_top) * top_offset_ratio)
    x1 = max(0, left_x - pad_x)
    x2 = min(img_w, right_x + pad_x)
    y1 = max(0, y_top + pad_top)
    y2 = min(img_h, y_chin)
    return x1, y1, x2, y2


# ========================
#  Face Mesh Completeness Check
# ========================
def check_mesh_completeness(landmarks, img_w, img_h):
    """
    Check if face mesh is complete (all landmarks visible in frame).
    Returns: (is_complete, visibility_ratio, out_of_bounds_count)
    """
    out_of_bounds = 0
    total_landmarks = len(landmarks)
    
    for lm in landmarks:
        x = lm.x * img_w
        y = lm.y * img_h
        # Check if landmark is outside frame boundaries
        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            out_of_bounds += 1
    
    visibility_ratio = 1.0 - (out_of_bounds / total_landmarks)
    is_complete = visibility_ratio >= MIN_LANDMARKS_VISIBLE
    
    return is_complete, visibility_ratio, out_of_bounds


def draw_face_mesh(frame, landmarks, img_w, img_h, color=(0, 255, 0), thickness=1):
    """
    Draw face mesh on the frame.
    """
    mp_face = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=1)
    
    # Draw the face mesh connections
    for connection in mp_face.FACEMESH_TESSELATION:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        
        start_x = int(start_point.x * img_w)
        start_y = int(start_point.y * img_h)
        end_x = int(end_point.x * img_w)
        end_y = int(end_point.y * img_h)
        
        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)


# ========================
#  Sunglasses/Occlusion Check
# ========================
def check_eye_occlusion(frame, landmarks, img_w, img_h):
    """
    Check if eyes are occluded by sunglasses or other objects.
    Returns: (eyes_visible, left_eye_brightness, right_eye_brightness)
    """
    # Left eye landmarks (approximate region)
    left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 144]
    # Right eye landmarks
    right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 373]
    
    def get_eye_region_brightness(eye_indices):
        points = []
        for idx in eye_indices:
            lm = landmarks[idx]
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Create mask for eye region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        # Extract eye region
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_region = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Calculate mean brightness (excluding zero pixels from mask)
        non_zero_pixels = eye_region[mask > 0]
        if len(non_zero_pixels) > 0:
            mean_brightness = np.mean(non_zero_pixels)
        else:
            mean_brightness = 0
        
        return mean_brightness
    
    left_brightness = get_eye_region_brightness(left_eye_indices)
    right_brightness = get_eye_region_brightness(right_eye_indices)
    
    # Check if both eyes are too dark (likely occluded)
    avg_eye_brightness = (left_brightness + right_brightness) / 2
    eyes_visible = avg_eye_brightness >= EYE_REGION_DARKNESS_THRESHOLD
    
    return eyes_visible, left_brightness, right_brightness


# ========================
#  Brightness Check
# ========================
def check_brightness(face_crop):
    """
    Check if face crop has acceptable brightness.
    Returns: (is_valid, reason, mean_brightness, overexposed_ratio)
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    overexposed_pixels = np.sum(gray > BRIGHTNESS_TOO_HIGH)
    total_pixels = gray.size
    overexposed_ratio = overexposed_pixels / total_pixels
    
    underexposed_pixels = np.sum(gray < BRIGHTNESS_TOO_LOW)
    underexposed_ratio = underexposed_pixels / total_pixels
    
    if overexposed_ratio > BRIGHTNESS_RATIO_THRESHOLD:
        return False, "TOO BRIGHT", mean_brightness, overexposed_ratio
    elif underexposed_ratio > BRIGHTNESS_RATIO_THRESHOLD:
        return False, "TOO DARK", mean_brightness, underexposed_ratio
    elif mean_brightness > 200:
        return False, "OVEREXPOSED", mean_brightness, overexposed_ratio
    elif mean_brightness < 60:
        return False, "UNDEREXPOSED", mean_brightness, underexposed_ratio
    else:
        return True, "OK", mean_brightness, overexposed_ratio


# ========================
#  Secure Classification
# ========================
def classify_with_confidence(probs, real_threshold=REAL_CONFIDENCE_THRESHOLD):
    """
    Classify with confidence threshold.
    """
    real_prob = probs[0][0].item()
    spoof_prob = probs[0][1].item()
    
    if real_prob >= real_threshold:
        return "REAL", (0, 255, 0), "PASS", real_prob, spoof_prob
    elif spoof_prob >= real_threshold:
        return "SPOOF", (0, 0, 255), "REJECT", real_prob, spoof_prob
    else:
        return "UNCERTAIN", (0, 165, 255), "REJECT", real_prob, spoof_prob


def multi_frame_decision(prediction_buffer, consistency_threshold=CONSISTENCY_THRESHOLD):
    """
    Make final decision based on multiple frames.
    """
    if len(prediction_buffer) < FRAME_BUFFER_SIZE:
        return "WARMING UP", (128, 128, 128), "WAIT"
    
    real_votes = sum(1 for p in prediction_buffer if p[0] == "REAL")
    spoof_votes = sum(1 for p in prediction_buffer if p[0] == "SPOOF")
    uncertain_votes = sum(1 for p in prediction_buffer if p[0] == "UNCERTAIN")
    
    total_frames = len(prediction_buffer)
    real_ratio = real_votes / total_frames
    
    if real_ratio >= consistency_threshold:
        return "✓ VERIFIED REAL", (0, 255, 0), "PASS"
    elif spoof_votes > uncertain_votes:
        return "✗ SPOOF DETECTED", (0, 0, 255), "REJECT"
    else:
        return "⚠ UNCERTAIN - DENIED", (0, 165, 255), "REJECT"


# ========================
#  Inference Pipeline
# ========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpoofNet(num_classes=2)
    model.load_state_dict(torch.load("normal-spoofnet.pth", map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded from normal-spoofnet.pth")
    print(f"🔒 Security Settings:")
    print(f"   - Real Confidence Threshold: {REAL_CONFIDENCE_THRESHOLD}")
    print(f"   - Brightness Range: {BRIGHTNESS_TOO_LOW}-{BRIGHTNESS_TOO_HIGH}")
    print(f"   - Multi-frame Verification: {ENABLE_MULTI_FRAME}")
    print(f"   - Mesh Completeness Check: {ENABLE_MESH_CHECK}")
    print(f"   - Sunglasses Detection: {ENABLE_SUNGLASSES_CHECK}")
    if ENABLE_MULTI_FRAME:
        print(f"   - Frame Buffer: {FRAME_BUFFER_SIZE} frames")
        print(f"   - Consistency Required: {CONSISTENCY_THRESHOLD*100}%")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False,
                                 max_num_faces=1,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    prediction_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

    print("[INFO] Starting secure webcam inference. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        display = frame.copy()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Check mesh completeness FIRST
            if ENABLE_MESH_CHECK:
                is_complete, visibility_ratio, out_of_bounds = check_mesh_completeness(landmarks, img_w, img_h)
                
                if not is_complete:
                    cv2.putText(display, f"⚠ INCOMPLETE FACE MESH", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    cv2.putText(display, f"Visibility: {visibility_ratio*100:.1f}% ({out_of_bounds} landmarks OOB)", 
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    cv2.rectangle(display, (10, 80), (250, 130), (0, 0, 200), -1)
                    cv2.putText(display, "✗ FACE NOT CENTERED", (20, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    prediction_buffer.clear()
                    cv2.imshow("Secure AntiSpoof System", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
            
            try:
                x1, y1, x2, y2 = mediapipe_crop_coords(landmarks, img_w, img_h)
                face_crop = frame[y1:y2, x1:x2]
                
                # Draw face mesh on display
                draw_face_mesh(display, landmarks, img_w, img_h, color=(0, 255, 0), thickness=1)
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

                if face_crop.size != 0:
                    # Check for sunglasses/occlusion
                    if ENABLE_SUNGLASSES_CHECK:
                        eyes_visible, left_bright, right_bright = check_eye_occlusion(frame, landmarks, img_w, img_h)
                        
                        if not eyes_visible:
                            cv2.putText(display, f"⚠ EYES OCCLUDED", (x1, y1 - 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            cv2.putText(display, f"Remove sunglasses/shades", (x1, y1 - 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            cv2.rectangle(display, (10, 10), (230, 60), (0, 0, 200), -1)
                            cv2.putText(display, "✗ OCCLUSION DETECTED", (20, 45),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            prediction_buffer.clear()
                            cv2.imshow("Secure AntiSpoof System", display)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            continue
                    
                    # Check brightness
                    brightness_ok, brightness_reason, mean_brightness, bright_ratio = check_brightness(face_crop)
                    
                    if not brightness_ok:
                        cv2.putText(display, f"⚠ {brightness_reason}", (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.putText(display, f"Brightness: {mean_brightness:.0f}", (x1, y1 - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.rectangle(display, (10, 10), (200, 60), (0, 0, 200), -1)
                        cv2.putText(display, "✗ LIGHTING ISSUE", (20, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        prediction_buffer.clear()
                        cv2.imshow("Secure AntiSpoof System", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue
                    
                    # All checks passed - proceed with model inference
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_tensor = preprocess(Image.fromarray(face_rgb)).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(face_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        
                        label, color, status, real_prob, spoof_prob = classify_with_confidence(probs)
                        
                        if ENABLE_MULTI_FRAME:
                            prediction_buffer.append((label, status))
                            final_label, final_color, final_status = multi_frame_decision(prediction_buffer)
                        else:
                            final_label = label
                            final_color = color
                            final_status = status
                        
                        # Display instant prediction
                        cv2.putText(display, f"Frame: {label}", (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(display, f"R:{real_prob:.2f} S:{spoof_prob:.2f}", (x1, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Display final decision
                        if ENABLE_MULTI_FRAME:
                            cv2.putText(display, final_label, (x1, y2 + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, final_color, 2)
                            
                            buffer_info = f"Buffer: {len(prediction_buffer)}/{FRAME_BUFFER_SIZE}"
                            cv2.putText(display, buffer_info, (x1, y2 + 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        else:
                            cv2.putText(display, final_label, (x1, y2 + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, final_color, 2)
                        
                        # Status indicator
                        status_bg = (0, 200, 0) if final_status == "PASS" else (0, 0, 200)
                        cv2.rectangle(display, (10, 10), (200, 60), status_bg, -1)
                        status_text = "✓ ACCESS GRANTED" if final_status == "PASS" else "✗ ACCESS DENIED"
                        cv2.putText(display, status_text, (20, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            except Exception as e:
                print("Face processing error:", e)
        else:
            cv2.putText(display, "No face detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.rectangle(display, (10, 60), (200, 110), (128, 128, 128), -1)
            cv2.putText(display, "⚠ NO FACE", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            prediction_buffer.clear()

        cv2.imshow("Secure AntiSpoof System", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("🔚 Inference stopped.")


if __name__ == "__main__":
    main()