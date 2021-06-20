import copy
import math


def main():
    results = "5_0_0_1_0_0_0_0_0_0_0_0_0_5_0_0_1_0_0_0_0>31_9_8_8_7_7_6_8_8_6_5_7_10_5_10_8_9_7_2_5_2>21_4_4_5_6_6_4_3_0_4_2_2_0_5_1_1_7_5_2_0_1>27_7_5_5_7_9_5_4_4_5_3_10_6_10_1_2_7_3_10_0_5>46_9_9_10_8_10_10_10_10_10_9_9_8_8_10_9_9_4_10_8_10>33_6_7_7_8_8_7_7_6_8_9_3_10_6_1_2_8_5_6_4_8>47_8_9_10_8_9_8_9_8_8_4_4_9_2_5_7_8_0_4_1_10>6_0_0_1_0_0_0_0_2_1_0_0_0_3_0_0_0_0_0_0_0>32_4_0_4_1_3_1_0_0_4_0_1_3_6_5_1_3_0_1_0_0>16_2_0_3_1_4_1_0_8_2_3_2_2_3_2_1_2_0_0_0_0>28_8_7_8_7_7_8_7_8_9_10_9_6_7_10_5_7_8_6_1_10>11_3_3_3_3_4_2_0_5_5_1_4_1_2_2_2_5_0_7_1_2>42_9_8_10_9_8_10_10_10_10_9_10_10_9_10_8_4_7_8_1_10>30_7_7_3_5_7_3_2_4_7_2_2_8_2_10_3_7_3_9_3_7>36_8_6_3_6_7_5_7_2_6_3_4_8_6_10_7_8_9_0_5_8>13_3_5_9_7_5_3_6_6_3_5_2_4_2_10_7_3_0_3_2_9>19_3_0_7_5_5_6_3_10_9_4_8_2_7_8_5_2_5_6_2_5>39_5_3_3_4_4_5_4_0_3_2_0_7_3_9_2_7_0_0_3_2>48_7_4_8_7_7_5_3_10_6_10_0_5_5_10_2_3_3_5_3_7>9_0_0_1_1_0_0_0_0_3_0_0_1_3_0_0_0_0_0_0_0>14_7_9_8_9_9_7_7_8_9_10_7_7_7_8_5_8_7_8_8_10>44_9_9_4_4_3_2_6_6_7_8_10_0_2_7_2_7_6_3_3_10>7_0_0_1_1_0_0_0_0_1_0_0_0_9_0_0_0_0_0_0_0>24_7_6_4_5_6_4_8_1_5_3_8_7_1_9_3_8_9_10_4_5>35_9_8_8_8_8_9_10_10_10_9_9_8_5_9_5_5_10_10_7_10>49_9_9_6_7_9_8_8_9_9_8_10_8_6_9_10_9_8_10_9_10>3_1_0_1_0_0_0_0_0_0_0_2_0_1_0_0_0_0_0_0_0>10_0_0_1_0_0_0_0_0_0_0_0_0_3_0_0_0_0_0_0_0>17_4_3_5_5_4_1_5_4_2_1_8_2_8_6_7_3_5_5_1_0>41_10_8_9_9_8_10_10_10_10_6_9_9_8_10_10_9_10_10_6_10>26_6_4_5_6_7_5_6_3_8_9_5_5_2_7_1_7_2_2_8_0>2_0_0_0_0_0_0_0_0_1_0_0_0_8_0_0_0_0_0_0_0>4_0_0_1_0_0_0_0_0_1_0_0_0_4_0_0_0_0_0_0_0>18_4_4_5_6_8_4_6_6_5_8_9_7_8_4_5_3_10_3_1_8>23_6_5_3_4_4_4_4_4_5_2_4_7_1_8_0_8_5_1_1_0>1_0_0_1_0_0_0_0_0_0_0_0_0_6_0_0_0_0_0_0_0>29_5_8_4_5_5_5_5_6_7_2_6_9_4_10_8_8_8_4_2_0>20_4_1_6_4_7_7_6_10_9_5_10_4_2_10_7_6_5_5_6_10>12_8_8_9_7_8_8_7_10_9_10_10_4_4_10_9_5_2_8_9_10>8_0_0_1_1_0_0_0_0_0_0_0_0_6_0_0_0_0_0_0_0>22_7_3_3_5_6_1_3_8_4_2_4_2_3_9_2_9_3_4_2_5>43_10_5_8_6_7_8_6_7_5_9_7_6_7_10_9_8_10_10_9_10>15_6_4_6_2_6_7_4_8_7_3_10_4_2_8_6_3_0_8_7_10>34_9_9_7_6_10_8_7_8_10_5_10_4_9_10_9_5_9_10_5_10>45_9_9_6_6_7_6_7_9_8_9_8_7_1_10_2_8_10_0_8_10>25_8_9_4_4_8_5_5_7_9_8_8_4_3_10_9_8_8_7_7_10>50_8_9_9_6_6_7_6_10_9_6_6_9_4_10_1_7_6_9_7_0>37_7_9_4_4_7_7_6_5_4_6_8_3_1_10_1_2_0_7_3_1>40_7_5_5_5_6_6_7_6_6_9_6_3_8_9_7_4_5_5_1_7>38_6_4_3_4_5_3_5_7_8_3_4_8_2_10_2_6_7_6_2_10"
    results_per_sentence = results.split(">")
    results_dict = {}
    for s in results_per_sentence:
        r = s.split("_")
        results_dict[int(r[0])] = r[1:]

    algorithm_results_dict = {
        "BIGRAM": [],
        "TRIGRAM": [],
        "BI_TRI_GRAM": [],
        "BI_TRI_GRAM_WEIGHTED": [],
        "HUMAN": []
    }
    algorithm_mean_dict = {
        "BIGRAM": 0,
        "TRIGRAM": 0,
        "BI_TRI_GRAM": 0,
        "BI_TRI_GRAM_WEIGHTED": 0,
        "HUMAN": 0
    }
    algorithm_variance_dict = copy.deepcopy(algorithm_mean_dict)
    algorithm_standard_deviation_dict = copy.deepcopy(algorithm_mean_dict)
    algorithm_scaled_mean_dict = copy.deepcopy(algorithm_mean_dict)
    algorithm_explained_variance_dict = copy.deepcopy(algorithm_mean_dict)

    for key in results_dict.keys():
        if key <= 10:
            algorithm_results_dict["BIGRAM"].extend(list(int(v) for v in results_dict[key] if v))
        elif key <= 20:
            algorithm_results_dict["TRIGRAM"].extend(list(int(v) for v in results_dict[key] if v))
        elif key <= 30:
            algorithm_results_dict["BI_TRI_GRAM"].extend(list(int(v) for v in results_dict[key] if v))
        elif key <= 40:
            algorithm_results_dict["BI_TRI_GRAM_WEIGHTED"].extend(list(int(v) for v in results_dict[key] if v))
        else:
            algorithm_results_dict["HUMAN"].extend(list(int(v) for v in results_dict[key] if v))

    for key in algorithm_results_dict.keys():
        results = algorithm_results_dict[key]
        algorithm_mean_dict[key] = sum(results) / len(results)
        differences_squared = list((algorithm_mean_dict[key] - v)**2 for v in results if v)
        algorithm_variance_dict[key] = sum(differences_squared) / len(differences_squared)
        algorithm_standard_deviation_dict[key] = math.sqrt(algorithm_variance_dict[key])

    perceived_human = algorithm_mean_dict["HUMAN"]
    scaler = 10 / perceived_human

    for key in algorithm_mean_dict:
        algorithm_scaled_mean_dict[key] = algorithm_mean_dict[key] * scaler
        algorithm_explained_variance_dict[key] = 1 - algorithm_variance_dict[key] / sum(algorithm_variance_dict.values())

    # print(len(algorithm_results_dict["BIGRAM"]) / 10)
    # print(algorithm_results_dict)
    print(algorithm_mean_dict)
    print(algorithm_scaled_mean_dict)
    print(algorithm_variance_dict)
    print(algorithm_explained_variance_dict)
    print(algorithm_standard_deviation_dict)


if __name__ == "__main__":
    main()