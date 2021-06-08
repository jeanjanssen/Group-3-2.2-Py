from ImageReader import ImageReader
import os
import cv2


def main():
    image_path_n = 'C:\\Users\\jeanj\\PycharmProjects\\Group-3-2.2-Py\\Negatives'
    image_path_p = 'C:\\Users\\jeanj\\PycharmProjects\\Group-3-2.2-Py\\Positives'
    image_list_n = os.listdir(image_path_n)
    image_list_p = os.listdir(image_path_p)
    ir = ImageReader()

    test = ir.convert_to_grayscale(image_list_p, image_path_p)
    cv2.imshow(test)


if __name__ == "__main__":
    main()
