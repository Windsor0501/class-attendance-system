import cv2
import os
import imutils
import numpy as np


def cv_show(label, img):
    cv2.namedWindow(label, cv2.WINDOW_NORMAL)
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# load face detector form given path
def load_detector(proto_path, model_path):
    face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    return face_detector

# return face location on original image; 
# 1.number of faces 2.face list 3.location list 4.confidence list
def face_detection(face_detector, image, confidence_default=0.5):
    # read image
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
        image = image
      
    # change size of image 
    (height, width) = image.shape[:2]
    image_resize = imutils.resize(image, width=900)
    
    # create the blobs
    image_blob = cv2.dnn.blobFromImage(
        cv2.resize(image_resize, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    face_detector.setInput(image_blob)
    detections = face_detector.forward()
    
    face_list = []
    locations_list = []
    confidence_list = []
    # image_copy = image.copy()

    # for all faces in th image
    for index in range(detections.shape[2]):

        confidence = detections[0, 0, index, 2]

        if confidence > confidence_default:

            locations = detections[0, 0, index, 3:7] * np.array([width, height, width, height])

            (startX, startY, endX, endY) = locations.astype('int')
            
            face = image[startY:endY, startX:endX]
            (face_height, face_width) = face.shape[:2]
            
            # cv2.rectangle(image_copy, (startX, startY), (endX, endY), (0, 255, 0), 5)
            
            face_list.append(face)
            locations_list.append((startX, startY, endX, endY))
            confidence_list.append(confidence)  
    
    return detections.shape[2], face_list, locations_list, confidence_list
