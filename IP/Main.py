from ImageReader import ImageReader
from LineFeature import LineFeature
from EdgeFeature import EdgeFeature
import time
import numpy as np


class WeakClassifier(object):
    def __init__(self, haar, theta, sign, weight):
        self.haar = haar
        self.theta = theta
        self.sign = sign
        self.weight = weight


def main():

    def predict(score, classifier):
        if score < classifier.theta:
            return -classifier.sign
        return classifier.sign

    def feature_weighted_error_rate(actual, predicted, weights):
        return sum(weights * (np.not_equal(actual, predicted)))


    #Preparing the training samples

    ir = ImageReader()
    image_path_n = 'C:\\Users\\Kaci\\Documents\\Group-3-face-recognition\\Negatives\\'
    image_path_p = 'C:\\Users\\Kaci\\Documents\\Group-3-face-recognition\\Positives\\'

    #image_list_n = ir.load_images_from_folder(image_path_n)
    #image_list_p = ir.load_images_from_folder(image_path_p)
    #gray_image_set_p = ir.convert_to_grayscale(image_list_p, image_path_p)
    #gray_image_set_n = ir.convert_to_grayscale(image_list_n)

    positive_samples = ir.get_positive_samples(image_path_p)
    negative_samples = ir.get_negative_samples(image_path_n)

    #positive_samples = []
    #negative_samples = []
    #for i in range(5):
     #   positive_samples.append(positive_s[i])
    #for i in range(5):
     #   negative_samples.append(negative_s[i])

    positive_samples = ir.convert_to_grayscale(positive_samples, image_path_p)
    negative_samples = ir.convert_to_grayscale(negative_samples, image_path_n)

    print("prepare")
    for i in range(len(positive_samples)):
        height = positive_samples[i].shape[0]
        width = positive_samples[i].shape[1]
        positive_samples[i] = ir.calculate_integral_image(height, width, positive_samples[i])

    for i in range(len(negative_samples)):
        height = negative_samples[i].shape[0]
        width = negative_samples[i].shape[1]
        negative_samples[i] = ir.calculate_integral_image(height, width, negative_samples[i])

    np.random.shuffle(positive_samples)
    np.random.shuffle(negative_samples)

    split = 0.95

    pos_split = int(len(positive_samples) * split)
    neg_split = int(len(negative_samples) * split)

    training_set = positive_samples[0:pos_split] + negative_samples[0:neg_split]
    testing_set = positive_samples[pos_split:] + negative_samples[neg_split:]
    training_set_integrals = []
    testing_set_integrals = []

    for i in range(len(testing_set)):
        testing_set_integrals.append(testing_set[i])

    for i in range(len(training_set)):
        training_set_integrals.append(training_set[i])

    pos_split = int(len(positive_samples) * split)
    neg_split = int(len(negative_samples) * split)
    nrPos = pos_split
    nrNeg = neg_split
    nrPos_test = len(positive_samples) - nrPos
    nrNeg_test = len(negative_samples) - nrNeg

    # Preparing the features we will us and train
    haar1 = EdgeFeature(200, 200, 1)
    haar2 = EdgeFeature(200, 200, 2)
    haar3 = EdgeFeature(200, 200, 3)
    haar4 = EdgeFeature(200, 200, 4)
    haar5 = LineFeature(200, 200, 1)
    haar6 = LineFeature(200, 200, 2)
    haar7 = LineFeature(200, 200, 3)
    haar8 = LineFeature(200, 200, 4)
    #haar9 = LineFeature(200, 50, 1)

    haar_feature_types = [haar1, haar2, haar3, haar4, haar5, haar6, haar7, haar8]
    features = []

    for i in range(len(haar_feature_types)):
        features.append(haar_feature_types[i])

    for i in range(len(training_set_integrals)):
        for j in range(len(features)):
            height_t = training_set_integrals[i].shape[0]
            width_t = training_set_integrals[i].shape[1]
            height_f = features[j].h
            width_f = features[j].w

            if height_f + 2 >= height_t:
                features[j].set_h(height_t - 2)
            if width_f + 2 >= width_t:
                features[j].set_w(width_t - 2)


    training_labels = [1] * nrPos + [-1] * nrNeg
    testing_labels = [1] * nrPos_test + [-1] * nrNeg_test
    print(training_labels)

    feature_weights = []
    weak_classifiers = []

    np.random.shuffle(features)

    errors = []
    scores = []
    thetas = []
    polarities = []

    # Apply features and record the average score for each feature on the images

    for j in features:
        #print("Polarity")
        avg_pos_score = 0.0
        avg_neg_score = 0.0
        for k in range(len(training_set)):
            score = j.get_score(1, 1, integral_im=training_set_integrals[k])
            scores.append(score)
            print("score :", score)
            if training_labels[k] == 1:
                avg_pos_score += score
            else:
                avg_neg_score += score

        avg_pos_score = avg_pos_score / nrPos
        avg_neg_score = avg_neg_score / nrNeg
        if avg_pos_score > avg_neg_score:
            polarity = 1
        else:
            polarity = -1
        polarities.append(polarity)

        theta = (avg_pos_score + avg_neg_score) / 2
        thetas.append(theta)

    print("Polarities :", polarities)
    print("Thetas :", thetas)
    # Create the Cascade
    F_target = 0.1
    f = 0.1
    f_i = 1
    F_i = 1
    cascade = []
    start_time = time.time()
    image_weights = [1.0/(2*nrPos)]*nrPos + [1.0/(2*nrNeg)]*nrNeg
    print("im_weights :", image_weights)
    show_stuff = False

    old_f = 0
    new_f = 0
    while F_i > F_target:
    #while f_i > f:
        print("train classifier--------------------------------------------------------------------------------------")
        ## Train classifier for stage i
        best_weak_classifier = 0
        lowest_error = float("inf")

        total = sum(image_weights)
        image_weights = [w / total for w in image_weights]
        TP = 0
        TN = 0

        f_i = 1
        cycle = 0

        # while f_i > f: # change condition TP>0.5 and TN>0.5 ?!
        while (TP / nrPos < 0.5) and (TN / nrNeg < 0.5):
        #while (TP > 0.5) and (TN > 0.5):
            print("long while loop, error for loop, find threshold")
            total = sum(image_weights)
            if total != 1:
                image_weights = [w / total for w in image_weights]

            # print(" ")
            errors = []
            # For every feature, find best threshold and compute corresponding weighted error
            loop_cnt = 0
            inner_loop_cnt = 0
            for j in features:
                #print("classifier object")
                # Create classifier object
                w_classifier = WeakClassifier(j, thetas[loop_cnt], polarities[loop_cnt], 0)

                # Compute weighted error
                predicted = []
                for sample in range(len(training_set)):
                    # Get predictions of all samples
                    score = scores[inner_loop_cnt]
                    predicted.append(predict(score, w_classifier))
                    inner_loop_cnt += 1

                weighted_error = feature_weighted_error_rate(training_labels, predicted, image_weights)
                errors.append(weighted_error)

                # Look for the lowest error and keep track of the corresponding classifier
                if weighted_error < lowest_error:
                    lowest_error = weighted_error
                    best_weak_classifier = w_classifier
                    # best_feature_index = features.index(j)

                loop_cnt += 1

            beta_t = lowest_error / (1 - lowest_error)

            if beta_t == 0:
                inverted_weight = 0
            else:
                inverted_weight = np.log(1 / beta_t)
            best_weak_classifier.weight = inverted_weight

            ## Update weights and evaluate current weak classifier ##
            predicted = []
            scores_debug = []
            for sample in range(len(training_set)):
                # Get weighted classification error
                score = best_weak_classifier.haar.get_score(1, 1, training_set_integrals[sample])
                scores_debug.append(score)
                predicted.append(predict(score, best_weak_classifier))

            FP = 0.0
            FN = 0.0
            TP = 0.0
            TN = 0.0
            colors_predicted = []
            for k in range(len(image_weights)):
                #print("labeling and predict")
                # if sample is correctly classified
                if training_labels[k] == 1 and predicted[k] == -1:
                    FN += 1
                if training_labels[k] == -1 and predicted[k] == 1:
                    FP += 1


                # Update image weights
                if training_labels[k] == predicted[k]:
                    image_weights[k] = image_weights[k] * beta_t
                    if predicted[k] == 1:
                        TP += 1
                    if predicted[k] == -1:
                        TN += 1
                if predicted[k] == -1:
                    colors_predicted.append('r')
                else:
                    colors_predicted.append('g')

            f_i = (FP / (2 * nrNeg)) + (FN / (2 * nrPos))
            print("TP/P :", TP / nrPos)
            print("TN/n :", TN / nrNeg)
            cycle += 1

        cascade.append(best_weak_classifier)

        strong_FP = 0.0
        strong_FN = 0.0

        cascade_scores = []
        cascade_colors_predicted = []
        for l in range(len(training_set)):
            #print("add labels")
            strong_score = 0.0
            for w_class in cascade:
                #print(w_class.weight)
                strong_score += w_class.weight * predict(
                    w_class.haar.get_score(1, 1, training_set_integrals[l]), w_class)
            cascade_scores.append(strong_score)
            clas = np.sign(strong_score)
            if clas == -1:
                cascade_colors_predicted.append('r')
            else:
                cascade_colors_predicted.append('g')

            if training_labels[l] == 1 and clas == -1:
                strong_FN += 1
            if training_labels[l] == -1 and clas == 1:
                strong_FP += 1

        print("strong fp :", strong_FP)
        F_i = (strong_FP / (2 * nrNeg)) + (strong_FN / (2 * nrPos))
        print("weird f value :", F_i)
        new_f = F_i
        if old_f == new_f:
            break
        old_f = new_f


    #Cascade list final
    print("len :", len(cascade))
    for j in cascade:
        print(j.haar.type)

    FP_test = 0.0
    FN_test = 0.0
    TP_test = 0.0
    TN_test = 0.0

    scores = []
    save = False

    for t in range(len(testing_set)):
        strong_score = 0.0
        for w_class in cascade:
            # print("Loc: " +str(w_class.haar.start))
            strong_score += w_class.weight * predict(
                w_class.haar.get_score(1, 1, testing_set[t]), w_class)
        clas = np.sign(strong_score)
        scores.append(strong_score)

        if testing_labels[t] == 1 and clas == -1:
            FN_test += 1

        if testing_labels[t] == -1 and clas == 1:
            FP_test += 1

        if testing_labels[t] == 1 and clas == 1:
            TP_test += 1

        if testing_labels[t] == -1 and clas == -1:
            TN_test += 1

    print("True Pos test :", TP_test, "/", nrPos_test)
    print("True Neg test :", TN_test, "/", nrNeg_test)
if __name__ == "__main__":
    main()
