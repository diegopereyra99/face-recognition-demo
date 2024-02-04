import time
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from imutils.video import FPS


# Load pre-trained MTCNN for face detection
mtcnn_detector = MTCNN(keep_all=True)

# Load pre-trained ResNet model for face recognition
resnet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Folder containing the database faces
database_folder = 'data/faces'

# Preprocess and save reference embeddings
reference_embeddings = {}
names = []
for filename in os.listdir(database_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(database_folder, filename)
        img = cv2.imread(img_path)
        
        # Detect face using MTCNN
        face = mtcnn_detector(img)[0]
        
        # Calculate embedding for the detected face
        embedding = resnet_model(face.unsqueeze(0)).detach().numpy()
        reference_embeddings[filename] = embedding
        names.append(filename.split(".")[0].capitalize())
        
ref_embeddings_arr = np.array(list(reference_embeddings.values()))

# Open camera for real-time face detection
cap = cv2.VideoCapture(2)
time.sleep(0.1)

# Initialize the FileVideoStream and start the FPS counter
# vs = FileVideoStream(cap).start()
fps = FPS().start()

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces
    batch_boxes, batch_probs = mtcnn_detector.detect(frame)
    
    if batch_boxes is not None and any(p > 0.9 for p in batch_probs):
        
        # Extract faces
        faces = mtcnn_detector.extract(frame, batch_boxes, None)

        for box, face, p in zip(batch_boxes, faces, batch_probs):
            x0, y0, x1, y1 = [int(b) for b in box]
            
            # Calculate embedding for the detected face
            detected_embedding = resnet_model(face.unsqueeze(0)).detach().numpy()

            # Compare with all reference embeddings in the database
            match_found = False
            similarities = np.inner(ref_embeddings_arr, detected_embedding)
            
            for filename, similarity in zip(names, similarities):
                print(f"Similarity with {filename}: {similarity.squeeze():.4f}")

            # You can set a threshold for similarity to decide if it's a match
            threshold = 0.5
            if (similarities > threshold).any():
                match_found = True
                name = names[similarities.argmax()]
                cv2.putText(frame, name, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # Draw bounding box based on match status
            color = (0, 255, 0) if match_found else (0, 0, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            # cv2.putText(frame, f"p={p:.2%}", (x0, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fps.update()
    
    cv2.imshow("Face Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
fps.stop()
print(f"Approximate FPS: {fps.fps():.2f}")

out.release()
cap.release()
cv2.destroyAllWindows()
