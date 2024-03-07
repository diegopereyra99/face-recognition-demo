import time
import shutil
from facenet_pytorch import InceptionResnetV1, MTCNN
import os
import numpy as np
import cv2
from PIL import Image
import torch
import csv


class FaceRecognizer(object):
    
    def __init__(self, match_threshold=0.5, detection_frame_size=320, data_folder="./data_fr") -> None:
        self.face_detector = MTCNN(keep_all=True, image_size=160)
        self.resnet_model = InceptionResnetV1(pretrained='vggface2').eval()
        self.detection_frame_size = detection_frame_size
        self.match_threshold = match_threshold
        self.reference_embeddings = {}
        # self.data_folder = data_folder
        self.csv_fname = f"data/csv/{time.strftime('%Y_%m_%d-%H_%M_%S')}.csv"
        os.makedirs(os.path.dirname(self.csv_fname), exist_ok=True)
        
        
    def load_database(self, database_path="./data/faces") -> None:
        for filename in os.listdir(database_path):
            img_path = os.path.join(database_path, filename)
            # img = cv2.imread(img_path)
            # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(Image.open(img_path))
            
            # Detect face using MTCNN
            face = self.extract_faces(img, resize_frame=False)[0]
            
            # Calculate embedding for the detected face
            embedding = self.resnet_model(face.unsqueeze(0)).detach().numpy()
            self.reference_embeddings[filename] = embedding.squeeze()
            
    def extract_faces(self, frame, align=False, return_bbox=False, resize_frame=True):
        # Assume frame is in RGB
        if resize_frame:
            size = frame.shape[:2]
            ratio = round(max(size) / self.detection_frame_size)
            det_size = (int(size[1] / ratio), int(size[0] / ratio))
            scaled_frame = cv2.resize(frame, det_size)
        else:
            ratio = 1
            scaled_frame = frame
            
        tic = time.time()*1000
        bboxes, probs, landmarks = self.face_detector.detect(scaled_frame, landmarks=True)
        if bboxes is not None:
            bboxes *= ratio
            landmarks *= ratio
            faces = self.face_detector.extract(frame, bboxes, None)
        else:
            bboxes = []
            faces = []
        
        toc2 = time.time()*1000
        # print(f"{toc2-tic:.2f} ms")
        
        if align and landmarks is not None:
            for i, lmark in enumerate(landmarks):
                face = torch2np(faces[i])
                aligned_face = align_face(
                    img=face, left_eye=lmark[0], right_eye=lmark[1]
                )
                
                faces[i] = np2torch(aligned_face)
        
        if return_bbox:
            return faces, bboxes
        else:
            return faces

    @property
    def ref_embeddings(self):
        return np.array(list(self.reference_embeddings.values()))

    @property
    def ref_names(self):
        return [fn.split(".")[0].capitalize() for fn in self.reference_embeddings]
    
    @property
    def ref_data(self):
        return {name: embedd for name, embedd in zip(self.ref_names, self.ref_embeddings)}
    
    def calculate_similarities(self, faces):
        face_embeddings = self.resnet_model(faces).detach().numpy()
        return np.matmul(face_embeddings, self.ref_embeddings.T)
    
    def find_matches(self, faces):
        similarities = self.calculate_similarities(faces)
        matches = []
        for s in similarities:
            if any(s > self.match_threshold):
                mtch = self.ref_names[s.argmax()]
            else:
                mtch = None
            matches.append(mtch)
            
        return matches
    
    def group_similarities(self, img_folder):
        filenames = [os.path.join(img_folder, fn) for fn in os.listdir(img_folder)]
        # it should be checked that there are less than n imgs (ex.: n=1000)
        raw_imgs = [np.array(Image.open(fn)) for fn in filenames]
        inp_tensor = torch.stack([np2torch(im.copy()) for im in raw_imgs])
        
        similarities = self.calculate_similarities(inp_tensor)
        
        return similarities
        
    def log_tracked_face(self, img_folder):
        
        similarities = self.group_similarities(img_folder)
        
        num_matches = (similarities > self.match_threshold).sum(axis=0)
        matches_count = dict(zip(self.ref_names, num_matches))
        
        info = {}
        full_date = os.listdir(img_folder)[0].split("-")[0]
        info["date"] = full_date[:10].replace("_", "-")
        info["hour"] = full_date[11:].replace("_", ":")
        
        if num_matches.sum() > 0:
            info["recognized_person"] = max(matches_count, key=matches_count.get)
            shutil.rmtree(img_folder)
        else:
            info["recognized_person"] = None

        info["num_detections"] = len(similarities)
        info["top1_num_recognitions"] = num_matches.max()
        info["total_recognitions"] = num_matches.sum()
        info["n_unique_recognitions"] = sum(num_matches > 0)
        info["object_id"] = os.path.basename(img_folder)
        
        file_exists = os.path.isfile(self.csv_fname)
        with open(self.csv_fname, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(info.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(info)
        
        print(info)

        

def align_face(img, left_eye, right_eye):
    """
    Align a given image horizantally with respect to their left and right eye locations
    Args:
        img (np.ndarray): pre-loaded image with detected face
        left_eye (list or tuple): coordinates of left eye with respect to the you
        right_eye(list or tuple): coordinates of right eye with respect to the you
    Returns:
        img (np.ndarray): aligned facial image
    """
    # if eye could not be detected for the given image, return image itself
    if left_eye is None or right_eye is None:
        return img

    # sometimes unexpectedly detected images come with nil dimensions
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img

    angle = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
    img = Image.fromarray(img)
    img = np.array(img.rotate(angle))
    return img

def torch2np(img):
    img = img.detach().numpy().transpose(1, 2, 0)
    return (img * 128 + 127.5).astype(np.uint8)

def np2torch(img):
    img = torch.Tensor(img.transpose(2, 0, 1))
    return (img - 127.5) / 128