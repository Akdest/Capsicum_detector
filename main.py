# main.py
import cv2
import numpy as np
import pandas as pd
import time
import asyncio
import base64
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket
from scipy.spatial import KDTree
from collections import Counter
import math

app = FastAPI()

# ==============================
# LOAD XKCD COLORS
# ==============================
colors_df = pd.read_csv("XKCDcolors_balanced.csv")
rgb_values = colors_df[['red', 'green', 'blue']].values
color_names = colors_df['colorname'].values
color_tree = KDTree(rgb_values)

def get_closest_color_name(rgb):
    _, idx = color_tree.query(rgb)
    return color_names[idx]

def normalize_capsicum_color(color):
    c = color.lower()
    if "green" in c: return "Green Capsicum"
    if "red" in c: return "Red Capsicum"
    if "yellow" in c: return "Yellow Capsicum"
    if "orange" in c: return "Orange Capsicum"
    return None

# ==============================
# LOAD YOLO MODEL
# ==============================
model = YOLO("best.pt")  # replace with your model path

# ==============================
# GLOBAL STORAGE
# ==============================
DIST_THRESH = 60
tracked_objects = {}  # id -> {"center":(x,y), "last_seen":time}
counter = Counter()
next_id = 0

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Camera not accessible")

# ==============================
# HELPER: OBJECT MATCHING
# ==============================
def is_same_object(sig, existing, dist_thresh=DIST_THRESH):
    sx, sy = sig
    # Defensive: ensure existing is a tuple/list with 2 elements
    if isinstance(existing, (list, tuple)) and len(existing) == 2:
        ex, ey = existing
    else:
        return False
    return math.hypot(sx - ex, sy - ey) < dist_thresh

# ==============================
# WEBSOCKET ENDPOINT
# ==============================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global next_id
    await ws.accept()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640,480))
        now = time.time()

        # YOLO detection
        results = model(frame, conf=0.4, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Center ROI for saturation
                h, w, _ = roi.shape
                cx1, cx2 = int(w*0.3), int(w*0.7)
                cy1, cy2 = int(h*0.3), int(h*0.7)
                center_roi = roi[cy1:cy2, cx1:cx2]
                if center_roi.size == 0:
                    continue

                hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
                if hsv[:,:,1].mean() < 40:
                    continue

                # Average color
                avg_bgr = center_roi.mean(axis=(0,1))
                avg_rgb = avg_bgr[::-1]
                raw_color = get_closest_color_name(avg_rgb)
                final_color = normalize_capsicum_color(raw_color)
                if final_color is None:
                    continue

                # Check if already tracked
                assigned_id = None
                for obj_id, obj in tracked_objects.items():
                    if is_same_object((cx,cy), obj["center"]):
                        assigned_id = obj_id
                        obj["center"] = (cx,cy)
                        obj["last_seen"] = now
                        break

                # New object
                if assigned_id is None:
                    assigned_id = next_id
                    tracked_objects[next_id] = {"center":(cx,cy), "last_seen":now}
                    next_id += 1
                    counter[final_color] += 1

                # Draw
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{final_color} ID:{assigned_id}",
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

        # Clean up old objects (not seen for 3 seconds)
        to_delete = [obj_id for obj_id,obj in tracked_objects.items() if now - obj["last_seen"] > 3.0]
        for obj_id in to_delete:
            del tracked_objects[obj_id]

        # Encode frame to base64
        _, jpeg = cv2.imencode('.jpg', frame)
        encoded = base64.b64encode(jpeg).decode()

        await ws.send_json({
            "frame": encoded,
            "counts": dict(counter)
        })

        await asyncio.sleep(0.03)
