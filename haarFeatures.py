import cv2
import os
import numpy as np
from numpy.lib import math

image_path = 'images\\'
image_list = os.listdir(image_path)

values = np.zeros((1, 1))
integral_image = np.zeros((1, 1))

for image in image_list:
    img = cv2.imread(image_path + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = 224
    width = gray.shape[1]*height/gray.shape[0]
    rounded_width = math.ceil(width)
    values.resize((height, rounded_width))
    integral_image.resize((height, rounded_width))
    gray = cv2.resize(gray, (int(width), height), None, 0.5, 0.5, interpolation=cv2.INTER_AREA)
    #cv2.imshow('img', gray)
    for i in range(height):
        for j in range(rounded_width-1):
            values[i, j] = gray[i, j]
    for k in range(height):
        sumM = 0
        for l in range(rounded_width):
            sumM = sumM + values[k, l]
            if k == 0:
                integral_image[k, l] = sumM
            else:
                integral_image[k, l] = integral_image[k - 1, l] + sumM

print(values)
print(integral_image)
