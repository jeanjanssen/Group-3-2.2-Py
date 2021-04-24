'''
OUTLINE:
Face emotion recognition
    Start off by loading pre-stored images

Live Web-cam support

We will be used Python, Open CV, and Deepface

DeepFace (for deep learning models implementation) - pre-trained models - pip install deepface
'''

# Let us do all of our imports
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace


def _main():
    bgr_img = cv2.imread('images/Baby.jpg')
    # rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    predictions = DeepFace.analyze(bgr_img)
    print (predictions)

    cv2.imshow('Emotion Detection Image', bgr_img)

    cv2.waitKey(0)
    cv2.destroyWindow('Friends')


if __name__ == '__main__':
    _main()
