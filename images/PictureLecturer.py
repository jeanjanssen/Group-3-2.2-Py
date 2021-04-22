import cv2
import os


def load_images_from_folder(folder):
    imagesArray = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            imagesArray.append(img)
    return imagesArray


images = load_images_from_folder(os.path.abspath(os.getcwd()))

for image in images:
    cv2.imshow('Picture', image)
    cv2.waitKey(0)
    cv2.destroyWindow('Picture')
