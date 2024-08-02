from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import cv2
import time
import sys
import tensorflow as tf
import FaceDetection_mtcnn
from FaceRecognizer import FaceRecognizer
import argparse


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    return session


def get_timeString():
    now_timestamp = time.time()
    now_structTime = time.localtime(now_timestamp)
    timeString_pattern = '%Y%m%d_%H%M%S'
    now_timeString_1 = time.strftime(timeString_pattern, now_structTime)
    now_timeString_2 = ('%.4f' %(now_timestamp%1))[2:]
    now_timeString = now_timeString_1 + '_' + now_timeString_2

    return now_timeString


def get_new_box(box, margin, image_size):
    image_width, image_height = image_size
    x1, y1, x2, y2 = box
    new_x1 = max(0, x1 - margin / 2)
    new_y1 = max(0, y1 - margin / 2)
    new_x2 = min(image_width, x2 + margin / 2)
    new_y2 = min(image_height, y2 + margin / 2)
    new_box = new_x1, new_y1, new_x2, new_y2

    return new_box


class FaceDetector(object):
    def __init__(self, model_dirPath='./resources/mtcnn_model'):
        self.session = get_session()
        with self.session.as_default():
            self.pnet, self.rnet, self.onet = FaceDetection_mtcnn.create_mtcnn(
                self.session, model_dirPath)

    # return location and features of face images
    def detect_image(self, image_3d_array, margin=8):
        min_size = 20
        threshold_list = [0.6, 0.7, 0.7]
        factor = 0.7
        box_2d_array, point_2d_array = FaceDetection_mtcnn.detect_face(
            image_3d_array, min_size,
            self.pnet, self.rnet, self.onet,
            threshold_list, factor)
        box_2d_array_1 = box_2d_array.reshape(-1, 5)
        box_2d_array_2 = box_2d_array_1[:, 0:4]
        box_list = []
        image_height, image_width, _ = image_3d_array.shape
        image_size = image_width, image_height
        for box in box_2d_array_2:
            new_box = get_new_box(box, margin, image_size)
            box_list.append(new_box)
        box_2d_array_3 = np.array(box_list).astype('int')
        if len(point_2d_array) == 0:
            point_2d_array_1 = np.empty((0, 10))
        else:
            point_2d_array_1 = np.transpose(point_2d_array, [1, 0])
        return box_2d_array_3, point_2d_array_1


def get_affine_image_3d_array(original_image_3d_array, box_1d_array, point_1d_array):
    affine_percent_1d_array = np.array([0.3333, 0.3969, 0.7867, 0.4227, 0.7, 0.7835])
    x1, y1, x2, y2 = box_1d_array
    clipped_image_3d_array = original_image_3d_array[y1:y2, x1:x2]
    clipped_image_width = x2 - x1
    clipped_image_height = y2 - y1
    clipped_image_size = np.array([clipped_image_width, clipped_image_height])
    old_point_2d_array = np.float32([
        [point_1d_array[0] - x1, point_1d_array[5] - y1],
        [point_1d_array[1] - x1, point_1d_array[6] - y1],
        [point_1d_array[4] - x1, point_1d_array[9] - y1]
    ])
    new_point_2d_array = (affine_percent_1d_array.reshape(-1, 2)
                          * clipped_image_size).astype('float32')
    affine_matrix = cv2.getAffineTransform(old_point_2d_array, new_point_2d_array)
    new_size = (112, 112)
    clipped_image_size = (clipped_image_width, clipped_image_height)
    affine_image_3d_array = cv2.warpAffine(clipped_image_3d_array, affine_matrix, clipped_image_size)
    affine_image_3d_array_1 = cv2.resize(affine_image_3d_array, new_size)

    return affine_image_3d_array


# face verify; multi-faces allowed
last_saveTime = time.time()
def get_personName_list(image_3d_array, box_2d_array, point_2d_array, face_recognizer):
    # global last_saveTime
    assert box_2d_array.shape[0] == point_2d_array.shape[0]
    personName_list = []
    for box_1d_array, point_1d_array in zip(box_2d_array, point_2d_array):
        affine_image_3d_array = get_affine_image_3d_array(
            image_3d_array, box_1d_array, point_1d_array)
        personName = face_recognizer.get_personName_1(affine_image_3d_array)
        personName_list.append(personName)
        '''
        interval = 1
        if time.time() - last_saveTime >= interval:
            last_saveTime = time.time()
            dirPath = './resources/affine_faces/' + personName
            if not os.path.isdir(dirPath):
                os.makedirs(dirPath)
            time_string = get_timeString()
            fileName = time_string + '.jpg'
            filePath = os.path.join(dirPath, fileName)
            image = Image.fromarray(affine_image_3d_array)
            image.save(filePath)
            '''
    return personName_list


import math
def get_drawed_image_3d_array(image_3d_array, box_2d_array, personName_list, point_2d_array,
                    show_box=True, show_personName=True,
                    show_keypoints=False, show_personQuantity=True):
    assert len(box_2d_array) == len(personName_list), '请检查函数的参数'
    person_quantity = len(box_2d_array)
    image_height, image_width, _ = image_3d_array.shape
    if person_quantity != 0:
        for index in range(person_quantity):
            if show_box:
                box = box_2d_array[index]
                x1, y1, x2, y2 = box
                leftTop_point = x1, y1
                rightBottom_point = x2, y2
                color = [255, 0, 0]
                thickness = math.ceil((image_width + image_height) / 500)
                cv2.rectangle(image_3d_array, leftTop_point, rightBottom_point, color, thickness)
            if show_personName:
                text = personName_list[index]
                image = Image.fromarray(image_3d_array)
                imageDraw = ImageDraw.Draw(image)
                fontSize = math.ceil((image_width + image_height) / 35)
                font = ImageFont.truetype('./font/calibril.ttf', fontSize, encoding='utf-8')
                textRegionLeftTop = (x1+5, y1)
                color = (255, 0, 0)
                imageDraw.text(textRegionLeftTop, text, color, font=font)
                image_3d_array = np.array(image)
            if show_keypoints:
                point_1d_array = point_2d_array[index]
                for i in range(5):
                    point = point_1d_array[i], point_1d_array[i+5]
                    radius = math.ceil((image_width + image_height) / 300)
                    color = (0, 255, 0)
                    thickness = -1
                    cv2.circle(image_3d_array, point, radius, color, thickness)
    if show_personQuantity:
        text = 'Total number of faces detected: %d' % person_quantity
        image = Image.fromarray(image_3d_array)
        imageDraw = ImageDraw.Draw(image)
        fontSize = math.ceil((image_width + image_height) / 50)
        font = ImageFont.truetype('calibri.ttf', fontSize, encoding='utf-8')
        textRegionLeftTop = (20, 20)
        imageDraw.text(textRegionLeftTop, text, (34, 139, 34), font=font)
        image_3d_array = np.array(image)
    return image_3d_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_keypoints', action='store_true', default=False)
    parser.add_argument('--show_personQuantity', action='store_true', default=True)
    parser.add_argument('--video_aviFilePath', default=None)
    parser.add_argument('--margin', type=int, default=8)
    argument_namespace = parser.parse_args()
    return argument_namespace



if __name__ == '__main__':
    argument_namespace = parse_args()
    show_keypoints = argument_namespace.show_keypoints # used in visualization
    show_personQuantity = argument_namespace.show_personQuantity # used in visualization
    video_aviFilePath = argument_namespace.video_aviFilePath # video_path; used in visualization
    margin = argument_namespace.margin # used in face detection
    
    cameraIndex = 0
    camera = cv2.VideoCapture(cameraIndex)
    windowName = "faceDetection_demo"
    
    is_successful, image_3d_array = camera.read()
    if is_successful:
        print('Loading FaceRecognizer successful')
        face_detector = FaceDetector() 
        face_recognizer = FaceRecognizer() 
        print('Initialize successful')
    else:
        print('Camera not access')
        sys.exit()

    if video_aviFilePath != None:
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
        image_height, image_width, _ = image_3d_array.shape
        image_size = image_width, image_height
        videoWriter = cv2.VideoWriter(video_aviFilePath, fourcc, 6, image_size)

    # loop here to capture images
    while is_successful:
        startTime = time.time()

        is_successful, bgr_3d_array = camera.read() 
        
        rgb_3d_array = cv2.cvtColor(bgr_3d_array, cv2.COLOR_BGR2RGB)
        
        box_2d_array, point_2d_array = face_detector.detect_image(rgb_3d_array, margin)
        usedTime = time.time() - startTime
        print('Consuming %.4f seconds to detect the face' %usedTime)
        
        startTime = time.time()
        if box_2d_array.shape[0] != 0:
            
            personName_list = get_personName_list(rgb_3d_array, box_2d_array, point_2d_array, face_recognizer)
            
            show_box=True
            show_personName=True
            drawed_image_3d_array = get_drawed_image_3d_array(rgb_3d_array, box_2d_array,
                personName_list, point_2d_array, show_box, show_personName,
                show_keypoints, show_personQuantity)
            bgr_3d_array = cv2.cvtColor(drawed_image_3d_array, cv2.COLOR_RGB2BGR)
            usedTime = time.time() - startTime
            print('Consuming %.4f seconds to draw the image' %usedTime)
        cv2.imwrite('test.jpg',bgr_3d_array)
        cv2.imshow(windowName, bgr_3d_array)
        
        if video_aviFilePath != None:
            videoWriter.write(bgr_3d_array)
        
        pressKey = cv2.waitKey(10)
        if 27 == pressKey or ord('q') == pressKey:
            cv2.destroyAllWindows()
            sys.exit()
    print('Process finished')
