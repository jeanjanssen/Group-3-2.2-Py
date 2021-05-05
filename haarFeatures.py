import cv2
import os
import numpy as np
from PIL import Image, ImageOps


image_path = 'images\\'
image_list = os.listdir(image_path)
gray_image_list = []

for im in image_list:
    img_gray = Image.open(image_path + im)
    gray_image = ImageOps.grayscale(img_gray)
    gray_image_list.append(gray_image)

# values = []
#
# for image in gray_image_list:
#     img = Image.open(image)
#     height = 224
#     width = img.shape[1]*height/img.shape[0]
#     img = cv2.resize(img, (int(width), height), None, 0.5, 0.5, interpolation=cv2.INTER_AREA)
#     cv2.imshow('img', img)
#     for i in height:
#         for j in width:
#             values = img.getpixel((i, j))
#             print(values)