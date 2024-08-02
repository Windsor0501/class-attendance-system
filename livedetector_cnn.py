import argparse
import inspect
import os
import sys
import cv2 
import imutils

import cv2 as cv
import numpy as np

from FaceDetection_res import load_detector, face_detection
from spf_tools import TorchCNN, VectorCNN, FaceDetector 


class LiveDetectorCNN():
    def __init__(self):
        self.spoof_path = os.path.normpath('./resources/spf_models/MN3_antispoof.xml')
        self.spoof_model = VectorCNN(self.spoof_path)

        self.proto_path = os.path.normpath('./resources/face_detection_model/deploy.prototxt')
        self.model_path = os.path.normpath('./resources/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')
        self.face_detector = load_detector(self.proto_path, self.model_path)


    # num_faces, face_list, location_list, confidence_list
    def face_detection_cnn(self, frame, confidence_default=0.15):
        return face_detection(self.face_detector, frame, confidence_default)

    
    # locations store the location of faces; could be replaced with other methods
    # output[k][1] store confidence of the face; threshold set to 0.4, the lower the model oriented to spoof
    def pred_spoof(self, frame, locations):
        """Get predictions for all detected faces on the frame"""
        faces = []
        for rect in locations:
            left, top, right, bottom = rect
            # cut face according coordinates of locations
            faces.append(frame[top:bottom, left:right])
        if faces:
            output = self.spoof_model.forward(faces)
            output = list(map(lambda x: x.reshape(-1), output))
            return output
        return None, None


    def draw_detections(self, frame, detections, confidence, thresh=0.4):
        """Draws detections and labels"""
        for i, rect in enumerate(detections):
            left, top, right, bottom = rect
            if confidence[i][1] > thresh:
                label = f'spoof: {round(confidence[i][1]*100, 3)}%'
                cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
            else:
                label = f'real: {round(confidence[i][0]*100, 3)}%'
                cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
            label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
            top = max(top, label_size[1])
            cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                        (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

        return frame


# for test only; invalid now
if __name__ == '__main__':
    spoof_path = os.path.normpath('./resources/spf_models/MN3_antispoof.xml')
   
    proto_path = os.path.normpath('./resources/face_detection_model/deploy.prototxt')
    model_path = os.path.normpath('./resources/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cameraIndex = 0
    camera = cv2.VideoCapture(cameraIndex, cv2.CAP_DSHOW)
    
    face_detector = load_detector(proto_path, model_path)
    spoof_model = VectorCNN(spoof_path)
    
    while True:
        is_successful, frame = camera.read()
        if not is_successful:
            break
          
        num_faces, face_list, location_list, confidence_list = face_detection(face_detector, frame, confidence_default=0.15)
        
        if num_faces > 0:
            confidence = pred_spoof(frame, location_list, spoof_model)
        
            frame = draw_detections(frame, location_list, confidence)
        
        cv.imshow('face detection', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    