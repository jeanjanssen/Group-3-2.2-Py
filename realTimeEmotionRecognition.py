"""
OUTLINE:
Face emotion recognition
    Start off by loading pre-stored images

Live Web-cam support

We will be used Python, Open CV, and Deepface

DeepFace (for deep learning models implementation) - pre-trained models - pip install deepface
"""

# Let us do all of our imports
import cv2
from deepface import DeepFace
import os

cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

def _main():
    # DONE FOR A TEST IMAGE
    """
    bgr_img = cv2.imread('images/Baby.jpg')
    # rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    predictions = DeepFace.analyze(bgr_img)
    print (predictions)

    cv2.imshow('Emotion Detection Image', bgr_img)

    cv2.waitKey(0)
    cv2.destroyWindow('Emotion Detection Image')
    """

    cap = cv2.VideoCapture(1)
    # Check if the webcam opened properly so the whole program does not crash
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError('Webcam cannot be opened')

    while True:
        ret, frame = cap.read()  # This allows us to read one image from a video
        result = DeepFace.analyze(frame, actions=['emotion'])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        # Drawing a rectangle around the actual face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Use putText() method for inserting text on video
        cv2.putText(frame, result['dominant_emotion'], (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
        cv2.imshow('Emotional Detection In Real Time', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    _main()
