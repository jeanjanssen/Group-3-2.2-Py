import os

import cv2
import sys

# Gets the name of the image file (filename) from sys.argv
# imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_alt.xml"
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
filename_list = []

# This creates the cascade classification from file
faceCascade = cv2.CascadeClassifier(cascPath)


# The image is read and converted to grayscale
# image = cv2.imread('TestImages/000001.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        filename_list.append(filename)
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
    # print("Detected {0} faces!".format(len(faces)))
    fpos = open(os.getcwd() + '\\tpos.txt', 'r')
    content_pos = fpos.readlines()
    fpos.close()
    fneg = open(os.getcwd() + '\\tneg.txt', 'r')
    content_neg = fneg.readlines()
    fneg.close()
    if len(faces) >= 1:
        if filename_list[i] + '\n' in content_pos:
            true_positives += 1
        else:
            false_positives += 1
    else:
        if filename_list[i] + '\n' in content_neg:
            true_negatives += 1
        else:
            false_negatives += 1

# This draws a green rectangle around the faces detected
# for (x, y, w, h) in faces:
#    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imshow("Faces Detected", image)
p = true_positives + false_negatives
n = true_negatives + false_positives
accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)
recall = true_positives / (true_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
b = 1
f_score = ((b**2 + 1) * precision * recall) / ((b**2 * precision) + recall)

print("Accuracy: ", accuracy)
print("Recall: ", recall)
print("Precision: ", precision)
print("F-score: ", f_score)
