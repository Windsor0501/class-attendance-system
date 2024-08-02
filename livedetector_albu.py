from pylab import imshow
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import seaborn as sns 
import pandas as pd
import os

from FaceDetection_res import load_detector, face_detection

import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

from datasouls_antispoof.pre_trained_models import create_model
from datasouls_antispoof.class_mapping import class_mapping


class LiveDetectorAlbu():
    def __init__(self):
        self.model = create_model("tf_efficientnet_b3_ns")
        self.transform = albu.Compose([albu.PadIfNeeded(min_height=400, min_width=400),
                            albu.CenterCrop(height=400, width=400), 
                            albu.Normalize(p=1), 
                            albu.pytorch.ToTensorV2(p=1)], p=1)
        
        self.proto_path = os.path.normpath('./resources/face_detection_model/deploy.prototxt')
        self.model_path = os.path.normpath('./resources/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')
        self.face_detector = load_detector(self.proto_path, self.model_path)
        self.model.eval()
        
    # num_faces, face_list, location_list, confidence_list
    def face_detection_albu(self, frame, confidence_default=0.15):
        return face_detection(self.face_detector, frame, confidence_default)
    
    
    def pred_proof(self, frame):
        with torch.no_grad():
            predictions = self.model(torch.unsqueeze(self.transform(image=frame)['image'], 0)).numpy()[0]
        return predictions


# for test only; invalid now
if __name__ == '__main__':
    proto_path = os.path.normpath('./resources/face_detection_model/deploy.prototxt')
    model_path = os.path.normpath('./resources/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cameraIndex = 0
    camera = cv2.VideoCapture(cameraIndex, cv2.CAP_DSHOW)
    
    face_detector = load_detector(proto_path, model_path)
    
    model = create_model("tf_efficientnet_b3_ns")
    model.eval()

    transform = albu.Compose([albu.PadIfNeeded(min_height=400, min_width=400),
                            albu.CenterCrop(height=400, width=400), 
                            albu.Normalize(p=1), 
                            albu.pytorch.ToTensorV2(p=1)], p=1)
    
    while True:
        is_successful, frame = camera.read()
        if not is_successful:
            break
          
        num_faces, face_list, location_list, confidence_list = face_detection(face_detector, frame, confidence_default=0.15)
        
        if len(location_list) > 0:
            with torch.no_grad():
                predictions = model(torch.unsqueeze(transform(image=frame)['image'], 0)).numpy()[0]
        
            left, top, right, bottom = location_list[0]
            if np.sum(predictions[1:4]) / 3 < 0.2:
                label = f'real: {round(predictions[0]*100, 3)}%'
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)    
            else:
                label = f'spoof: {round(predictions[0]*100, 3)}%'
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            top = max(top, label_size[1])
            cv2.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                        (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            
            
        cv2.imshow('face detection', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
