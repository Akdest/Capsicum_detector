import cv2
import numpy as np
import pandas as pd
import threading
import tkinter as tk
from ultralytics import YOLO
from collections import Counter
from scipy.spatial import KDTree

# ==============================
# Load XKCD Color Dataset
# ==============================
colors_df = pd.read_csv("XKCDcolors_balanced.csv")
colors_df = colors_df[['red', 'green', 'blue', 'colorname']]

rgb_values = colors_df[['red', 'green', 'blue']].values
color_names = colors_df['colorname'].values
color_tree = KDTree(rgb_values)

def get_closest_color_name(rgb):
    _, idx = color_tree.query(rgb)
    return color_names[idx]

def normalize_capsicum_color(color):
    c = color.lower()
    if "green" in c:
        return "Green Capsicum"
    if "red" in c:
        return "Red Capsicum"
    if "yellow" in c:
        return "Yellow Capsicum"
    if "orange" in c:
        return "Orange Capsicum"
    return None

# ==============================
# Load YOLO ONNX Model
# ==============================
model = YOLO(r"D:\jupyter projects\color_det\capsicum.v1i.yolov8\runs\detect\train\weights\best.onnx")

# ==============================
# Camera
# ==============================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

# ==============================
# Shared Data (IMPORTANT)
# ==============================
counter = Counter()
counted_track_ids = set()

running = True

# ==============================
# ------------------------------
# TKINTER DASHBOARD
# ------------------------------
# ==============================
root = tk.Tk()
root.title("Capsicum Live Counter Dashboard")
root.geometry("300x300")

label_title = tk.Label(root, text="Live Capsicum Count", font=("Arial", 14))
label_title.pack(pady=10)

text_box = tk.Text(root, font=("Arial", 12))
text_box.pack(fill=tk.BOTH, expand=True)

def update_dashboard():
    text_box.delete("1.0", tk.END)

    if len(counter) == 0:
        text_box.insert(tk.END, "No detections yet...\n")
    else:
        for k, v in counter.items():
            text_box.insert(tk.END, f"{k} : {v}\n")

    root.after(500, update_dashboard)  # auto refresh every 0.5 sec

update_dashboard()

# ==============================
# MAIN DETECTION LOOP
# ==============================
def detection_loop():
    global running

    print("📸 Running YOLO + Tracking + Dashboard... Press 'q' to stop")

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model.track(frame, persist=True, conf=0.4, verbose=False)

        for r in results:
            if r.boxes.id is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                h, w, _ = roi.shape
                cx1, cx2 = int(w * 0.3), int(w * 0.7)
                cy1, cy2 = int(h * 0.3), int(h * 0.7)
                center_roi = roi[cy1:cy2, cx1:cx2]

                if center_roi.size == 0:
                    continue

                hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
                if hsv[:, :, 1].mean() < 40:
                    continue

                avg_bgr = center_roi.mean(axis=(0, 1))
                avg_rgb = avg_bgr[::-1]

                raw_color = get_closest_color_name(avg_rgb)
                final_color = normalize_capsicum_color(raw_color)

                if final_color is None:
                    continue

                if track_id not in counted_track_ids:
                    counter[final_color] += 1
                    counted_track_ids.add(track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {track_id} | {final_color}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Capsicum Detection Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# ==============================
# THREAD FOR DETECTION
# ==============================
threading.Thread(target=detection_loop, daemon=True).start()

# ==============================
# START TKINTER LOOP
# ==============================
root.mainloop()

running = False