import cv2
import os
import numpy as np
from numpy.lib import math


class ImageRead:
    image_path_n = 'Negatives\\'
    image_path_p = 'Positives\\'
    image_list_n = os.listdir(image_path_n)
    image_list_n = os.listdir(image_path_p)
    values = np.zeros((1, 1))

    def convert_to_grayscale(self, image_list, image_path):
        for image in image_list:
            img = cv2.ImageRead(image_path + image)
            grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height = 224
            width = grayscale_image.shape[1] * height / grayscale_image.shape[0]
            rounded_width = math.ceil(width)
            ImageRead.values.resize((height, rounded_width))
            grayscale_image = cv2.resize(grayscale_image, (int(width), height), None, 0.5, 0.5,
                                         interpolation=cv2.INTER_AREA)

        return grayscale_image

    def calculate_integral_image(self, height, width, grayscale_image):
        integral_array = []
        integral_image = np.zeros((height, width))

        for i in range(height):
            for j in range(width - 1):
                ImageRead.values[i, j] = grayscale_image[i, j]
        for k in range(height):
            sum = 0
            for l in range(width):
                sum = sum + ImageRead.values[k, l]
                if k == 0:
                    integral_image[k, l] = sum
                else:
                    integral_image[k, l] = integral_image[k - 1, l] + sum

        integral_array.append(ImageRead.values)
        return integral_array
