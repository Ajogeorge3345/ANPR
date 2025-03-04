from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *
import easyocr
import firebase_admin
from firebase_admin import credentials, db, firestore
import datetime
import os

# Initialize Firebase
cred = credentials.Certificate('pass.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://anpr-d05b8-default-rtdb.asia-southeast1.firebasedatabase.app/'
})
dbStore = firestore.client()

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

# Load video
cap = cv2.VideoCapture("./videos/parking_data9.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Load YOLO model
model = YOLO("./Yolo-Weights/best_l.pt")
classNames = ["number_plate"]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [10, 400, 1280, 450]
# limits = [100, 500, 1280, 550]

# Firebase database reference
ref = db.reference('/')
totalCount = []
active_plates = []
frame_count = 0

# Disable GUI on Windows
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  

# Flag to control display
headless_mode = False  # Set to False to enable GUI display

while True:
    ret, img = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "number_plate" and conf > 0.3:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    resultsTracker = tracker.update(detections)
    
    text = "No plate"  # Default value
    
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        
        # Ensure valid coordinates (defensive coding)
        if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and x1 < img.shape[1] and y1 < img.shape[0]:
            roi = img[y1:y2, x1:x2]
            
            if roi.size > 0:  # Check if ROI is not empty
                # Resize for better OCR
                scale_factor = 3
                resized_image = cv2.resize(roi, (roi.shape[1] * scale_factor, roi.shape[0] * scale_factor))
                gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                
                try:
                    # Apply OCR to detect plate text
                    plate = reader.readtext(resized_image)
                    
                    if plate and len(plate) > 0:
                        text = plate[0][1]
                        cvzone.putTextRect(img, text, (x1, y1 - 30), scale=1, thickness=1, colorR=(0, 255, 0))
                        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=5, rt=2)
                except Exception as e:
                    print(f"OCR error: {e}")
                    continue

            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    if text in active_plates:
                        dbStore.collection('left_vehicles').document(text).set({'plate_number': text, 'Out': timestamp})
                        plates_ref = ref.child('active_plates')
                        snapshot = plates_ref.order_by_child('plate_number').equal_to(text).get()
                        for key in snapshot.keys():
                            plates_ref.child(key).delete()
                        active_plates.remove(text)
                        print(f"Vehicle exit: {text} at {timestamp}")
                    else:
                        ref.child('active_plates').push().set({'plate_number': text, 'timestamp': timestamp})
                        dbStore.collection('detected_plates').document(text).set({'plate_number': text, 'In': timestamp})
                        active_plates.append(text)
                        print(f"Vehicle entry: {text} at {timestamp}")

    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50), scale=2, thickness=2, colorR=(0, 255, 255))
    
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames. Current vehicle count: {len(totalCount)}")
    
    # Try to display only if not in headless mode
    if not headless_mode:
        try:
            cv2.imshow("Detected Plates", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            print("Warning: Display not available. Running in headless mode.")
            headless_mode = True  # Switch to headless mode if display fails
    
    # Break condition without relying on GUI
    if frame_count >= 1000:  # Limit to prevent endless processing
        print("Reached maximum frame count")
        break

cap.release()
try:
    cv2.destroyAllWindows()
except:
    pass

print(f"Processing complete. Total vehicles detected: {len(totalCount)}")