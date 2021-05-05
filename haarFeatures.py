import cv2
import os
from numpy.lib import math

image_path = 'images\\'
image_list = os.listdir(image_path)

values = []

for image in image_list:
    img = cv2.imread(image_path + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = 224
    width = gray.shape[1]*height/gray.shape[0]
    rounded_width = math.ceil(width)
    gray = cv2.resize(gray, (int(width), height), None, 0.5, 0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('img', gray)
    for i in range(height):
        for j in range(rounded_width-1):
            new_value = gray[i, j]
            values.append(new_value)