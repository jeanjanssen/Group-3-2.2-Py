import os

import cv2
import sys

# Gets the name of the image file (filename) from sys.argv
# imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_alt.xml"
true_positives = 0
false_negatives = 0

# This creates the cascade classification from file
faceCascade = cv2.CascadeClassifier(cascPath)


# The image is read and converted to grayscale
# image = cv2.imread('TestImages/000001.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


image_list = load_images_from_folder("C:\\Users\\jeanj\\PycharmProjects\\Group-3-2.2-Py\\AccuracyTest\\TestImages")

# The face or faces in an image are detected
# This section requires the most adjustments to get accuracy on face being detected.
for i in range(len(image_list)):
    image = image_list[i]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print("Detected {0} faces!".format(len(faces)))
    if len(faces) > 0:
        true_positives = true_positives + 1
    else:
        false_negatives = false_negatives + 1

# This draws a green rectangle around the faces detected
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#cv2.imshow("Faces Detected", image)
#cv2.waitKey(0)

print(true_positives)
print(false_negatives)
