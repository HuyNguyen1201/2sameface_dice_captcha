import base64
from PIL import Image
import cv2
from io import BytesIO
import numpy as np
import cv2
import random
import os
import string
from model.yolov5.detect_custom import run as yolo_run
from tensorflow.keras.models import load_model
from skimage import transform
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

_YOLO_WEIGHT_PATH = "./model/yolov5/runs/train/exp/weights/best.pt"
_CNN_WEIGHT_PATH = "./model/generated_dataset_no_crop_100_100_4.h5"

def convert_base64(base64_str, path=None):
    # read image
    im = Image.open(BytesIO(base64.b64decode(base64_str)))
    # random filename
    folder_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    if path:
        folder_path = os.path.join(path, folder_name)
    else:
        folder_path = folder_name
    # make new foler
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, folder_name + ".png" )
    # save image
    im.save(file_path,'PNG')
    return file_path,folder_path

def read_split_image(file_path, folder_path):
    sample_size = 100
    im = cv2.imread(file_path)
    # split to 6 samples
    counter = 0
    for row  in range(2):
        for column in range(3):
            counter += 1
            anchor = [row*sample_size, column*sample_size]
            im_croped = im[anchor[0]:anchor[0] + sample_size, anchor[1]:anchor[1]+sample_size]
            sample_path = os.path.join(folder_path,str(counter) + ".png")
            cv2.imwrite(sample_path, im_croped)
    # delete original image
    os.remove(file_path)

def get_new_image(position, image):
    img_size = 40
    sample_size = 100
    side_1 = img_size//2
    side_2 = img_size-side_1-1
    center_x = (position[0]+position[2])//2
    center_y = (position[1]+position[3])//2
    if center_x - side_1<=0:
        center_x = side_1
    if center_x + side_2 > sample_size:
        center_x = sample_size-side_2
    if center_y - side_1<=0:
        center_y = side_1
    if center_y + side_2 > sample_size:
        center_y = sample_size-side_2
    return image[center_y-side_1:center_y+side_2,center_x-side_1:center_x+side_2]

def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')
    np_image = transform.resize(np_image,(100,100,3))
    np_image = np.expand_dims(np_image, axis = 0)
    return np_image

def run(data):
    base64_str = data['base64image']
    # read and crop sample
    file_path,folder_path = convert_base64(base64_str)
    read_split_image(file_path, folder_path)
    # crop dice's face <YOLO>
    r = yolo_run(weights=_YOLO_WEIGHT_PATH, source=folder_path, imgsz=[100,100], line_thickness=1)
    croped_im_path = []
    for result in r:
        bouding_box = result[1]
        if len(bouding_box) < 2: continue
        bouding_box_1 = bouding_box[0][2]
        bouding_box_2 = bouding_box[1][2]
        name = result[0]
        # crop 2 image
        im = cv2.imread(os.path.join(folder_path,name + '.png'))
        im_1 = get_new_image(bouding_box_1,im)
        im_2 = get_new_image(bouding_box_2, im)
        horizontal_im = cv2.hconcat([im_1,im_2])
        im_path = os.path.join(folder_path, name + '.png')
        cv2.imwrite(im_path, horizontal_im)
        croped_im_path.append(im_path)
    # predict by CNN
    model = load_model(_CNN_WEIGHT_PATH)
    result_predict = []
    for path in croped_im_path:
        im = load(path)
        result  = model.predict(im)
        result_predict.append(result)
    return result_predict.index(max(result_predict)) + 1
    