import cv2
import dlib
import imutils
import time
import sys
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist

import argparse


class LiveDetector():
    def __init__(self):
        self.shape_predictor_path = './resources/dlib_model/shape_predictor_68_face_landmarks.dat'
        
        # load dlib model for landmark detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor_path)
        # thresholds for different detectors
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 2
        self.MOUTH_AR_THRESH = 0.5
        self.MOUTH_AR_CONSEC_FRAMES = 3
        # counter for different detectors
        self.blink_frame_counter = 0
        self.blink_counter = 0
        self.blink_flag = False
        
        self.mouth_frame_counter = 0
        self.mouth_counter = 0
        self.mouth_flag = False
        
        self.nod_counter = 0
        self.nod_flag = False


    # reset when successfully detected
    def reset(self):
        self.blink_frame_counter = 0
        self.blink_counter = 0
        self.blink_flag = False
    
        self.mouth_frame_counter = 0
        self.mouth_counter = 0
        self.mouth_flag = False
        
        self.nod_counter = 0
        self.nod_flag = False
        
        
    def eye_aspect_ratio(self, eye):
        # compute the distance between two eyes in vertical dimension
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the distance between two eyes in  dimension
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    
    def mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[2] - mouth[9])
        B = np.linalg.norm(mouth[4] - mouth[7])
        C = np.linalg.norm(mouth[0] - mouth[6])
        mar = (A + B) / (2.0 * C)
        return mar

      
    def is_point_above_jaw(self, jaw, point):
        for i in range(len(jaw) - 1):
            if self.is_point_above_line(jaw[i], jaw[i + 1], point):
                return False
        return True

    def is_point_above_line(self, point1, point2, point):
        x1, y1 = point1
        x2, y2 = point2
        x, y = point
        return (y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1 > 0
    
      
    # return blink frame counter and blink counter; set zero if successfully detected
    def blink_detector_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector(gray, 0)
        
        for face in faces:
            shape = self.predictor(gray, face)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            left_eye = shape[42:48]
            right_eye = shape[36:42]

            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            if left_ear < self.EYE_AR_THRESH and right_ear < self.EYE_AR_THRESH:
                self.blink_frame_counter += 1
            else:
                if self.blink_frame_counter >= self.EYE_AR_CONSEC_FRAMES:
                    self.blink_counter += 1
                    self.blink_flag = True
        
        return self.blink_counter, self.blink_flag 

    
    def mouth_detector_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector(gray, 0)
        
        for face in faces:
            shape = self.predictor(gray, face)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            mouth = np.matrix(shape[48:68])

            mouth_ratio = self.mouth_aspect_ratio(mouth)
            if mouth_ratio > self.MOUTH_AR_THRESH:
                self.mouth_frame_counter += 1
            else:
                if self.mouth_frame_counter >= self.MOUTH_AR_CONSEC_FRAMES:
                    self.mouth_counter += 1
                    self.mouth_flag = True
                   
        return self.mouth_counter, self.mouth_flag    


    # invalid now
    def nodding_detector_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector(gray, 0)
        for face in faces:
            shape = self.predictor(gray, face)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            jaw = shape[0:17]
            nose = shape[27:36]
            
            nose_center = (nose[3][0], nose[3][1] - 15)
            if self.is_point_above_jaw(jaw, nose_center):
                self.nod_counter += 1
                self.nod_flag = True
                
        return self.nod_counter, self.nod_flag
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_keypoints', action='store_true', default=False)
    parser.add_argument('--show_personQuantity', action='store_true', default=True)
    parser.add_argument('--video_aviFilePath', default=None)
    parser.add_argument('--margin', type=int, default=8)
    argument_namespace = parser.parse_args()
    return argument_namespace


if __name__ == '__main__':
    font = cv2.FONT_HERSHEY_SIMPLEX
    cameraIndex = 0
    camera = cv2.VideoCapture(cameraIndex)
    
    live_detector = LiveDetector()
     
    while True:
        
        is_successful, frame = camera.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if not is_successful:
            break  
        
        counter, flag = live_detector.mouth_detector_frame(frame)
        
        if flag:
            print('Mouth detection successes')
            cv2.putText(frame, "MouthNum: " + str(counter), (20, 80), font, 1, (255,20,147), 1, cv2.LINE_AA)
            live_detector.reset()
        else:
            print('Mouth detection fails')
        
        cv2.namedWindow("camera", 0)
        cv2.imshow("camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    camera.release()
    cv2.destroyAllWindows()
    