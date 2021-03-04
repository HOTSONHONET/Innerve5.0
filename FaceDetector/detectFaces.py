import numpy as np
import cv2
import os


class DetectMyFace:
    def __init__(self, prototxt_path = "./deploy.prototxt", caffeModel_path = "./res10_300x300_ssd_iter_140000.caffemodel"):
        self.prototxt_file = prototxt_path
        self.caffeModel = caffeModel_path
        self.face_detector = cv2.dnn.readNet(self.prototxt_file, self.caffeModel)
    
    def detect_faces(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400), (104, 177, 123))

        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        locs = []; confidences = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])

                (startx, starty, endx, endy) = box.astype("int")

                startx, starty = max(0, startx), max(0, starty)
                endx, endy = min(w-1, endx), min(h-1, endy)

                locs.append((startx, starty, endx, endy))
                confidences.append(confidence)
        
        locs = np.array(locs)
        confidences = np.array(confidences, dtype = "int")
        return locs


    def __str__(self):
        return "DetectMyFace"
