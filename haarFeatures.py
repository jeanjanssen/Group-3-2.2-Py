import cv2
import os
import numpy as np
from numpy.lib import math

image_path = 'images\\'
image_list = os.listdir(image_path)

matrixList = []
integral_image = np.zeros((1, 1))

for image in image_list:
    values = np.zeros((1, 1))
    img = cv2.imread(image_path + image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = 224
    width = gray.shape[1] * height / gray.shape[0]
    rounded_width = math.ceil(width)
    values.resize((height, rounded_width))
    integral_image.resize((height, rounded_width))
    gray = cv2.resize(gray, (int(width), height), None, 0.5, 0.5, interpolation=cv2.INTER_AREA)
    # cv2.imshow('img', gray)
    for i in range(height):
        for j in range(rounded_width - 1):
            values[i, j] = gray[i, j]
    for k in range(height):
        sumM = 0
        for l in range(rounded_width):
            sumM = sumM + values[k, l]
            if k == 0:
                integral_image[k, l] = sumM
            else:
                integral_image[k, l] = integral_image[k - 1, l] + sumM

    matrixList.append(values)


# print(values)
# print(integral_image)


def edge_vertical(x, y, w, h):
    coordsVertical = [(x, y), (x + w / 2, y), (x + w, y), (x, y - h), (x + w / 2, y - h), (x + w, y - h)]
    mat_e_vert = np.zeros((w, h))
    for i in range(h):
        for j in range(w):
            if j < w / 2:
                mat_e_vert[i, j] = 0
            else:
                mat_e_vert[i, j] = 1
    return mat_e_vert


def edge_horizontal(x, y, w, h):
    coordsHorizontal = [(x, y), (x + w, y), (x, y - h / 2), (x, y - h), (x + w, y - h / 2), (x + w, y - h)]
    mat_e_hor = np.zeros((w, h))
    for i in range(h):
        for j in range(w):
            if i < h / 2:
                mat_e_hor[i, j] = 0
            else:
                mat_e_hor[i, j] = 1
    return mat_e_hor


def line_vertical(x, y, w, h):
    coordsLineVert = [(x, y), (x + math.floor(w / 3), y), (x + 2 * math.floor(w / 3), y), (x + w, y), (x, y - h),
                      (x + math.floor(w / 3), y - h), (x + 2 * math.floor(w / 3), y - h), (x + w, y - h)]
    mat_l_vert = np.zeros((w, h))
    for i in range(h):
        for j in range(w):
            if j < math.floor(w / 3) or j >= 2 * math.floor(w / 3):
                mat_l_vert[i, j] = 0
            else:
                mat_l_vert[i, j] = 1
    return mat_l_vert


def line_horizontal(x, y, w, h):
    coordsLineHor = [(x, y), (x + w, y), (x, y - math.floor(h / 3)), (x, y - 2 * math.floor(h / 3)), (x, y - h),
                     (x + w, y - math.floor(h / 3)), (x + w, y - 2 * math.floor(h / 3)), (x + w, y - h)]
    mat_l_hor = np.zeros((w, h))
    for i in range(h):
        for j in range(w):
            if i < math.floor(h / 3) or i >= 2 * math.floor(h / 3):
                mat_l_hor[i, j] = 0
            else:
                mat_l_hor[i, j] = 1
    return mat_l_hor


test = line_vertical(0, 0, 6, 6)
print(test)
