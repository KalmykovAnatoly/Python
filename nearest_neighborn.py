import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 5000)


def k_nearest_neighbors(data, predict=None, k=3):
    if len(data) >= 3:
        warnings.warn('K is set to the value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence


accuracies = []

for i in range(5):
    df = pd.read_csv("breast-cancer-wisconsin.data.txt")
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(len(full_data) * test_size)]
    test_data = full_data[-int(len(full_data) * test_size):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in train_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    print('Accuracy:', correct / total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))