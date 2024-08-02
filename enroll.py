from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import cv2
import time
import sys
import tensorflow as tf
import FaceDetection_mtcnn
import argparse


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def get_new_box(box, margin, image_size):
    image_width, image_height = image_size
    x1, y1, x2, y2 = box
    new_x1 = max(0, x1 - margin/2)
    new_y1 = max(0, y1 - margin/2)
    new_x2 = min(image_width, x2 + margin/2)
    new_y2 = min(image_height, y2 + margin/2)
    new_box = new_x1, new_y1, new_x2, new_y2
    return new_box


class FaceDetector(object):
    def __init__(self,model_dirPath = './resources/mtcnn_model'):
        self.session = get_session()
        with self.session.as_default():
            self.pnet,self.rnet,self.onet = FaceDetection_mtcnn.create_mtcnn(
                self.session,model_dirPath)

    def detect_image(self, image_3d_array, margin=8):
        min_size = 20
        threshold_list = [0.6, 0.7, 0.7]
        factor = 0.7
        box_2d_array, point_2d_array = FaceDetection_mtcnn.detect_face(
            image_3d_array, min_size,self.pnet, self.rnet, self.onet,threshold_list, factor)
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
    """
    :param original_image_3d_array:     original image 
    :param box_1d_array:                faces box
    :param point_1d_array:              feature point for faces
    :return:                            face image
    """
    affine_percent_1d_array = np.array([0.3333, 0.3969, 0.7867, 0.4227, 0.7, 0.7835])
    x1, y1, x2, y2 = box_1d_array
    clipped_image_3d_array = original_image_3d_array[y1:y2, x1:x2]
    clipped_image_width = x2 - x1
    clipped_image_height = y2 - y1
    clipped_image_size = np.array([clipped_image_width, clipped_image_height])
    old_point_2d_array = np.float32([
        [point_1d_array[0]-x1, point_1d_array[5]-y1],
        [point_1d_array[1]-x1, point_1d_array[6]-y1],
        [point_1d_array[4]-x1, point_1d_array[9]-y1]
        ])
    new_point_2d_array = (affine_percent_1d_array.reshape(-1, 2)
        * clipped_image_size).astype('float32')
    affine_matrix = cv2.getAffineTransform(old_point_2d_array, new_point_2d_array)
    new_size = (112, 112)
    clipped_image_size = (clipped_image_width, clipped_image_height)
    affine_image_3d_array = cv2.warpAffine(clipped_image_3d_array,
        affine_matrix, clipped_image_size)
    affine_image_3d_array_1 = cv2.resize(affine_image_3d_array, new_size)
    return affine_image_3d_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dirPath', type=str, default='./resources/person_database')
    parser.add_argument('--out_dirPath', type=str, default='./resources/face_database')
    parser.add_argument('--margin', type=int, default=8)
    argument_namespace = parser.parse_args()
    return argument_namespace


if __name__ == '__main__':
    with open('./static/tmp/tmp_text.txt', 'w') as f:
        f.write('Enrollment Start!')
    argument_namespace = parse_args()
    in_dirPath = argument_namespace.in_dirPath
    out_dirPath = argument_namespace.out_dirPath
    margin = argument_namespace.margin
    face_detetor = FaceDetector()
    sub_dirName_list = next(os.walk(in_dirPath))[1]
    for sub_dirName in sub_dirName_list:
        in_sub_dirPath = os.path.join(in_dirPath, sub_dirName)
        out_sub_dirPath = os.path.join(out_dirPath, sub_dirName)
        fileName_list = next(os.walk(in_sub_dirPath))[2]
        print('File name:', fileName_list) 
        time.sleep(5) # time.sleep() here to avoid 'run_out_of_memory' error 
        for fileName in fileName_list:
            in_filePath = os.path.join(in_sub_dirPath, fileName)
            out_filePath = os.path.join(out_sub_dirPath, fileName)
            image_3d_array = np.array(Image.open(in_filePath))
            box_2d_array, point_2d_array = face_detetor.detect_image(image_3d_array, margin)
            face_quantity = len(box_2d_array)
            if face_quantity > 1 or face_quantity == 0:
                print('File %s has faces %d' %(in_filePath, face_quantity))
                continue
            box_1d_array = box_2d_array[0]
            point_1d_array = point_2d_array[0]
            affine_image_3d_array = get_affine_image_3d_array(image_3d_array,
                box_1d_array, point_1d_array)
            affine_image = Image.fromarray(affine_image_3d_array)
            if not os.path.isdir(out_sub_dirPath):
                os.makedirs(out_sub_dirPath)
            affine_image.save(out_filePath)
    print('Enrollment Finished!')
    with open('./static/tmp/tmp_text.txt', 'w') as f:
        f.write('Enrollment finished for all students!')
