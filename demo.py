import cv2
import numpy as np
from recognizer import FaceRecognizer

reco = FaceRecognizer()
reco.load_database()

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    original = frame.copy()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces, boxes = reco.extract_faces(rgb, return_bbox=True)
    
    if len(boxes) > 0:
            
        matches = reco.find_matches(faces)
        # matches = [None] * len(boxes)

        for box, face, match in zip(boxes, faces, matches):
            x0, y0, x1, y1 = [int(b) for b in box]
            # ff = face.detach().numpy().transpose(1, 2, 0)[:, :, ::-1]
            # frame[:160, :160] = (ff * 128 + 127.5).astype(np.uint8)
            
            if match is not None:
                cv2.putText(frame, match, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw bounding box based on match status
            color = (0, 0, 255) if match is None else (0, 255, 0)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            # cv2.putText(frame, f"p={p:.2%}", (x0, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()