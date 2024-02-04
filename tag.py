import cv2
from facenet_pytorch import MTCNN
import os

# Load pre-trained MTCNN for face detection
mtcnn_detector = MTCNN()

# Open camera for real-time face detection
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    original = frame.copy()

    # Detect faces using MTCNN
    boxes, _ = mtcnn_detector.detect(frame[None])
    boxes = boxes[0] # Remove batch dimension
    
    if boxes is not None:
        
        single_face = len(boxes) == 1
        color = (0, 255, 0) if single_face else (0, 0, 255)
        for box in boxes:
            box = [int(b) for b in box]
            x0, y0, x1, y1 = box
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        
        text = "Press 'T' to tag this person" if single_face else "More than one face"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.imshow("Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if (key == ord('t') or key == ord('T')) and single_face:
            # # Stop video capture
            cap.release()
            cv2.destroyAllWindows()

            # Input name from the user
            name = input('\033[1;37m' + 'Enter the name of the person: ' + '\033[0m')

            # Create the 'data/faces' directory if it doesn't exist
            save_dir = 'data/faces'
            os.makedirs(save_dir, exist_ok=True)

            # Save the current image with the entered name
            save_path = os.path.join(save_dir, f'{name}.jpg')
            cv2.imwrite(save_path, original)
            break

        elif key == ord('q') or key == ord('Q'):
            break

    else:
        cv2.imshow("Face Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if cv2.waitKey(1) & 0xFF == ord('q') or key == ord('Q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
