import cv2
import os

# ================= CONFIG =================
ROOT_DIR = "."  
OUTPUT_ROOT = "../FULL_DATASET_FRAMES"

FPS_TO_EXTRACT = 2
CONF_THRESHOLD = 0.6

CROP_TOP = 0.1
CROP_BOTTOM = 0.1
CROP_LEFT = 0.02
CROP_RIGHT = 0.02

TARGET_SIZE = 224  # Resize to 224x224

FACE_MODEL = "opencv_face_detector_uint8.pb"
FACE_CONFIG = "opencv_face_detector.pbtxt"
# =========================================

video_exts = (".mp4", ".avi", ".mov", ".mkv")

def clamp(val, minv, maxv):
    return max(minv, min(val, maxv))

net = cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_CONFIG)

splits = ["train", "test"]
classes = ["real_video", "attack"]

for split in splits:
    for cls in classes:

        input_dir = os.path.join(ROOT_DIR, split, cls)
        output_dir = os.path.join(OUTPUT_ROOT, split, cls)

        os.makedirs(output_dir, exist_ok=True)

        for video_name in os.listdir(input_dir):
            if not video_name.lower().endswith(video_exts):
                continue

            video_path = os.path.join(input_dir, video_name)
            video_id = os.path.splitext(video_name)[0]

            out_dir = os.path.join(output_dir, video_id)
            os.makedirs(out_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / FPS_TO_EXTRACT))

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_idx = 0
            saved_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                blob = cv2.dnn.blobFromImage(
                    frame, 1.0, (300, 300),
                    (104.0, 177.0, 123.0),
                    swapRB=False,
                    crop=False
                )

                net.setInput(blob)
                detections = net.forward()

                best_det = None
                best_conf = 0

                for i in range(detections.shape[2]):
                    conf = detections[0, 0, i, 2]
                    if conf > CONF_THRESHOLD and conf > best_conf:
                        best_conf = conf
                        best_det = detections[0, 0, i, 3:7]

                if best_det is None:
                    frame_idx += 1
                    continue

                x1 = int(best_det[0] * width)
                y1 = int(best_det[1] * height)
                x2 = int(best_det[2] * width)
                y2 = int(best_det[3] * height)

                w = x2 - x1
                h = y2 - y1

                x1 += int(CROP_LEFT * w)
                x2 -= int(CROP_RIGHT * w)
                y1 += int(CROP_TOP * h)
                y2 -= int(CROP_BOTTOM * h)

                x1 = clamp(x1, 0, width)
                x2 = clamp(x2, 0, width)
                y1 = clamp(y1, 0, height)
                y2 = clamp(y2, 0, height)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    frame_idx += 1
                    continue

                # ✅ Resize to 224x224
                face_resized = cv2.resize(face, (TARGET_SIZE, TARGET_SIZE))

                fname = f"img_{saved_idx:05d}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), face_resized)

                saved_idx += 1
                frame_idx += 1

            cap.release()
            print(f"✅ {split}/{cls}/{video_name} → {saved_idx} frames")

print("🎯 All videos processed successfully.")
