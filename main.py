import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import cv2
import numpy as np
from recognizer import FaceRecognizer, torch2np
from norfair import Tracker, Detection, draw_tracked_objects
from concurrent.futures import ThreadPoolExecutor
import shutil


# TODO: - Load database from cache
# - Integrate Tracker to FaceRecognizer
# - Adjust distance_threshold parameter (Use normalization with frame size?)


reco = FaceRecognizer()
reco.load_database()

source = 1
# source = "rtsp-video.mp4"
# source = "rtsp://admin:TripleClasico@192.168.0.110:554/Streaming/Channels/1"
cap = cv2.VideoCapture(source)

window_name = 'Face Detection'
# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.moveWindow(window_name, 1921, 0)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


tracker = Tracker(distance_function="euclidean", distance_threshold=300, hit_counter_max=10)
dying_objects = []

tracked_faces_dir = 'data/tracked_faces'
if os.path.exists(tracked_faces_dir):
    shutil.rmtree(tracked_faces_dir)

# Only 1 worker to not overload PC
with ThreadPoolExecutor(max_workers=1) as executor:
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame, restarting video capture")
            cap.release()
            # time.sleep(0.1)
            cap = cv2.VideoCapture(source)
            continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, boxes = reco.extract_faces(rgb, return_bbox=True)
        
        detections = [Detection(np.array([(x0, y0), (x1, y1)])) for (x0, y0, x1, y1) in boxes]
        tracked_objects = tracker.update(detections=detections)
        draw_tracked_objects(frame, tracked_objects, id_thickness=2)
        
        if len(boxes) > 0:
            
            for obj in tracked_objects:
                bbox = obj.last_detection.points
                ix = abs(boxes - bbox.flatten()).sum(axis=1).argmin()
                face = torch2np(faces[ix])[:, :, ::-1]
                obj_folder = os.path.join(tracked_faces_dir, f"{obj.id}")
                os.makedirs(obj_folder, exist_ok=True)
                save_path = os.path.join(obj_folder, f"{time.strftime('%Y_%m_%d_%H_%M_%S')}-{obj.age}.png")
                cv2.imwrite(save_path, face)
                

            for box in boxes:
                # print(box)
                x0, y0, x1, y1 = [int(b) for b in box]
                
                color = (0, 255, 0)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                # cv2.putText(frame, f"p={p:.2%}", (x0, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
                
        dead_objects = [obj for obj in dying_objects if obj not in tracked_objects]
        dying_objects = [obj for obj in tracked_objects if obj.hit_counter == 0]
        
        if len(dead_objects) > 0:
            for obj in dead_objects:
                obj_folder = os.path.join(tracked_faces_dir, f"{obj.id}")
                # find the best match for all the saved faces of that object
                executor.submit(reco.log_tracked_face, obj_folder) # Run this in parallel
                # reco.log_tracked_face(obj_folder)
                

        # print((time.time() - start) *1000)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()