import pandas
import math
from sklearn.neighbors import KNeighborsClassifier


def jaccards_index(training_row, test_row):
    p = 0
    q = 0
    r = 0
    for i in range(0, len(training_row)):
        if training_row[i] == 1 and test_row[i] == 1:
            p = p+1
        elif training_row[i] == 1:
            q = q+1
        elif test_row[i] == 1:
            r = r+1
    return (q+r)/(p+q+r)


def euclidian_distance(training_row, test_row):
    distance = 0.0
    for i in range(0, len(training_row)):
        distance += (int(training_row[i]) - int(test_row[i])) ** 2
    return math.sqrt(distance)


def majority(tab, training_classes):
    if len(tab) == 1:
        return training_classes.values[tab]
    count = 0
    for i in range(len(tab)):
        if training_classes.values[tab[i]] == 1:
            count = count + 1
    if count > math.floor(len(tab) / 2):
        return 1
    else:
        return 0


def main():
    df_test = pandas.read_excel('Testing dataset.xlsx')
    df_training = pandas.read_excel('Training dataset.xlsx')
    training_classes = df_training['Grade class 1: 90+  0:90-']
    df_training = df_training.drop(columns=['Grade class 1: 90+  0:90-'])
    df_test = df_test.drop(columns=['Grade class 1: 90+  0:90-'])
    df_prediction = pandas.read_csv('predictions.csv', delimiter=';')

    for index, row in df_test.iterrows():
        distances = []
        for index2, row2 in df_training.iterrows():
            distances.append([jaccards_index(row2.drop(['Unnamed: 0']), row.drop(['Unnamed: 0'])), index2])
        distances.sort()
        print(distances[:7])
        for k in [1, 3, 5, 7]:
            neighbours = []
            for i in range(0, k):
                neighbours.append(distances[i][1])
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(df_training.drop(columns=['Unnamed: 0']), training_classes.values)
            df_prediction['Prediction k=' + str(k)][index] = majority(neighbours, training_classes)
            df_prediction['scikit k=' + str(k)][index] = neigh.predict(row.drop(['Unnamed: 0']).values.reshape(1, -1))
    df_prediction.to_csv('result.csv')


if __name__ == "__main__":
    main()
