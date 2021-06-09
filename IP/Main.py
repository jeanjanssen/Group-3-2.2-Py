from ImageReader import ImageReader
from LineFeature import LineFeature
from EdgeFeature import EdgeFeature
import numpy as np


def main():
    ir = ImageReader()
    image_path_n = 'C:\\Users\\jeanj\\PycharmProjects\\Group-3-2.2-Py\\Negatives'
    image_path_p = 'C:\\Users\\jeanj\\PycharmProjects\\Group-3-2.2-Py\\Positives'

    image_list_n = ir.load_images_from_folder(image_path_n)
    image_list_p = ir.load_images_from_folder(image_path_p)
    gray_image_set_p = ir.convert_to_grayscale(image_list_p, image_path_p)
    gray_image_set_n = ir.convert_to_grayscale(image_list_n)

    def predict(score, classifier):
        if score < classifier.theta:
            return -classifier.sign
        return classifier.sign

    haar1 = EdgeFeature(200, 200, 1)
    haar2 = EdgeFeature(200, 200, 2)
    haar3 = EdgeFeature(200, 200, 3)
    haar4 = EdgeFeature(200, 200, 4)
    haar5 = LineFeature(200, 200, 1)
    haar6 = LineFeature(200, 200, 2)
    haar7 = LineFeature(200, 200, 3)
    haar8 = LineFeature(200, 200, 4)

    haar_feature_types = [haar1, haar2, haar3, haar4, haar5, haar6, haar7, haar8]
    features = []

    for i in range(len(haar_feature_types)):
        features.append(haar_feature_types[i])

    training_set = []
    training_set_integrals = []
    for i in range(len(training_set)):
        height = len(training_set[i][0])
        width = len(training_set[i])
        training_set_integrals.append(ir.calculate_integral_image(height, width, training_set[i]))

    feature_weights = []
    weak_classifiers = []

    np.random.shuffle(features)

    errors = []
    scores = []
    thetas = []
    polarities = []

    for j in features:
        avg_pos_score = 0.0
        avg_neg_score = 0.0
        for k in range(len(training_set)):
            score = features[i].get_score(1, 1, integral_im=training_set_integrals[k])


if __name__ == "__main__":
    main()
