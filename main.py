from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import cv2
import time
import sys
import torch
import math

# face recognition
from FaceRecognizer import FaceRecognizer
from verify import FaceDetector, get_personName_list 
# live detection for blink and mouth-open
from livedetector_blink import LiveDetector
# live detection with cnn
from livedetector_cnn import LiveDetectorCNN
# live detection with albu
from livedetector_albu import LiveDetectorAlbu

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp_path', default='./static/tmp/tmp_image.jpg') # path to save tmp image 
    parser.add_argument('--margin', type=int, default=8) # used in face detection
    parser.add_argument('--method', default='albu', help='blink, mouth-open, cnn, albu') 
    parser.add_argument('--frame_threshold', type=int, default=10) # used in live detection 
    argument_namespace = parser.parse_args()
    return argument_namespace


def draw_image(frame, box_list, personName_list, point_list, live_flag):
    number = len(box_list)
    height, width, _ = frame.shape
    color = (0, 0, 255) if live_flag else (255, 0, 0)
    for index in range(number):
        # box drawing
        box = box_list[index]
        x1, y1, x2, y2 = box
        leftTop_point = x1, y1
        rightBottom_point = x2, y2
        thickness = math.ceil((width + height) / 500)
        cv2.rectangle(frame, leftTop_point, rightBottom_point, color, thickness)
        # text drawing
        text = personName_list[index]
        image = Image.fromarray(frame)
        imageDraw = ImageDraw.Draw(image)
        fontSize = math.ceil((width + height) / 35)
        font = ImageFont.truetype('./font/calibril.ttf', fontSize, encoding='utf-8')
        textRegionLeftTop = (x1 + 5, y1)
        imageDraw.text(textRegionLeftTop, text, color, font=font)
        frame = np.array(image)
    
    text = 'Live detection succeeds' if live_flag else 'Live detection fails'
    image = Image.fromarray(frame)
    imageDraw = ImageDraw.Draw(image)
    fontSize = math.ceil((width + height) / 50)
    font = ImageFont.truetype('calibri.ttf', fontSize, encoding='utf-8')
    textRegionLeftTop = (20, 20)
    imageDraw.text(textRegionLeftTop, text, (34, 139, 34), font=font)
    frame = np.array(image)
    
    return frame
        

if __name__ == '__main__':
    with open('./static/tmp/tmp_text.txt', 'w') as f:
        f.write('Verification Start!')
    args = parse_args()
    tmp_path = args.tmp_path
    margin = args.margin
    method = args.method
    frame_threshold = args.frame_threshold
    
    cameraIndex = 0
    camera = cv2.VideoCapture(cameraIndex)
    
    frame_counter = 0
    if method == 'blink' or method == 'mouth-open':
        live_detector = LiveDetector()
        frame_threshold = 2 # suit blink and mouth-open detect
    elif method == 'cnn':
        live_detector = LiveDetectorCNN()
    elif method == 'albu':
        live_detector = LiveDetectorAlbu()
        frame_threshold = 2
    else:
        print('Method not finished')
        sys.exit()
    
    face_detector = FaceDetector() 
    face_recognizer = FaceRecognizer() 
    is_successful, image_3d_array = camera.read()
    
    if is_successful:
        print('Camera available')
    else:
        print('Camera not access')
        sys.exit()
    
    
    live_flag, last_flag = False, False
    print('Use {} for live detection'.format(method))
    while is_successful:
        is_successful, frame = camera.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # get face information of the image  
        box_list, point_list = face_detector.detect_image(frame_rgb, margin)
        
        if box_list.shape[0] != 0:
            #verification 
            personName_list = get_personName_list(frame_rgb, box_list, point_list, face_recognizer) # ['Bob', 'Pelosi', ...]
            
            tmp_flag = False 
            if method == 'blink':
                _, tmp_flag = live_detector.blink_detector_frame(frame)
            elif method == 'mouth-open':
                _, tmp_flag = live_detector.mouth_detector_frame(frame)
            elif method == 'cnn':
                num_faces, face_list, location_list, confidence_list = live_detector.face_detection_cnn(frame)
                if len(location_list) > 0:
                    confidences = live_detector.pred_spoof(frame, location_list)
                    tmp_flag = True if confidences[0][1] < 0.4 else False # one face only for live detection
            elif method == 'albu':
                num_faces, face_list, location_list, confidence_list = live_detector.face_detection_albu(frame)
                if len(location_list) > 0:
                    with torch.no_grad():
                        predictions = live_detector.pred_proof(frame)
                    # print(predictions)
                    tmp_flag = True if np.sum(predictions[1:4]) < 0.4 else False
            
            if tmp_flag:
                frame_counter += 1
                if method == 'blink' or method == 'mouth-open':
                    live_detector.reset() # reset parameters for live detector
            if frame_counter >= frame_threshold:
                live_flag = True

            frame = draw_image(frame, box_list, personName_list, point_list, live_flag)
        
        cv2.imwrite(tmp_path, frame)
        # cv2.imshow('test', frame) # for debug only
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if not last_flag and live_flag:
            print('Live detection succeeds with detection method {}'.format(method))
            detection_text = 'Live detection succeeds with detection method {}.'.format(method)
            for name in personName_list:
                print('Student ID {} has successfully signed in'.format(name))
                id_text = 'Student ID {} has successfully signed in.'.format(name)
            output = detection_text + id_text
            with open('./static/tmp/tmp_text.txt', 'w') as f:
                f.write(output)
            last_flag = live_flag
            live_flag = False
            frame_counter = 0
            
        else:
            last_flag = False

    camera.release()
    cv2.destroyAllWindows()
    