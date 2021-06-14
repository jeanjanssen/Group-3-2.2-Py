import cv2
import os
import numpy as np
from numpy.lib import math
import os
import glob


class ImageReader:
    def __init__(self):
        self.values = np.zeros((1, 1))

    def load_images_from_folder(self, folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
        return images

    def get_positive_samples(self, path):
        images = [cv2.imread(file) for file in glob.glob(path + "*.jpg")]
        return images

    def get_negative_samples(self, path):
        images = [cv2.imread(file) for file in glob.glob(path + "*.jpg")]
        return images

    def convert_to_grayscale(self, image_list, image_path):
        grayscale_image_list = []
        for image in image_list:
            #img = cv2.imread(image_path + image)
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height = 224
            width = grayscale_image.shape[1] * height / grayscale_image.shape[0]
            rounded_width = math.ceil(width)
            self.values.resize((height, rounded_width))
            grayscale_image = cv2.resize(grayscale_image, (int(width), height), None, 0.5, 0.5,
                                         interpolation=cv2.INTER_AREA)
            grayscale_image_list.append(grayscale_image)

        return grayscale_image_list

    def calculate_integral_image(self, height, width, grayscale_image):
        integral_array = []
        integral_image = np.zeros((height, width))
        temp = np.zeros((height, width))
        for i in range(height):
            for j in range(width - 1):
                temp[i, j] = grayscale_image[i, j]
        for k in range(height):
            sum = 0
            for l in range(width):
                sum = sum + temp[k, l]
                if k == 0:
                    integral_image[k, l] = sum
                else:
                    integral_image[k, l] = integral_image[k - 1, l] + sum

        #integral_array.append()
        return integral_image
