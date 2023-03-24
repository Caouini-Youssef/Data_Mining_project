import random

import pandas
import math
from sklearn.neighbors import KNeighborsClassifier
import numpy


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

def KFoldCrossValidation(df_validate, df_train, training_classes, fold, df_kfold):
    TP, FP, FN, TN = [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]
    for k in [1, 3, 5, 7]:
        TP[k] = 0
        FP[k] = 0
        FN[k] = 0
        TN[k] = 0
    for index, row in df_validate.iterrows():
        distances = []
        for index2, row2 in df_train.iterrows():
            distances.append([jaccards_index(row2.drop(['Unnamed: 0']), row.drop(['Unnamed: 0'])), index2])
        distances.sort()
        print(distances[:7])

        for k in [1, 3, 5, 7]:
            neighbours = []
            for i in range(0, k):
                neighbours.append(distances[i][1])
            prediction = majority(neighbours, training_classes)
            if (prediction == 1 and training_classes[index] == 1):
                TP[k] += 1
            elif (prediction == 1):
                FP[k] += 1
            elif (training_classes[index] == 1):
                FN[k] += 1
            else:
                TN[k] += 1
    for i in [1, 3, 5, 7]:
        df_kfold['K='+str(i)+'/'+'TP'][fold] = TP[k]
        df_kfold['K='+str(i)+'/'+'FP'][fold] = FP[k]
        df_kfold['K='+str(i)+'/'+'FN'][fold] = FN[k]
        df_kfold['K='+str(i)+'/'+'TN'][fold] = TN[k]


def main():
    main_df = pandas.read_excel('Full Wine Data.xlsx')
    main_df = numpy.array_split(main_df, 5)
    df_test = main_df.pop(random.randint(0,4))
    df_training = pandas.concat(main_df)
    df_test.reset_index(inplace=True)
    df_training.reset_index(inplace=True)

    training_classes = df_training['Grade class 1: 90+  0:90-']
    df_training = df_training.drop(columns=['Grade class 1: 90+  0:90-'])
    df_test = df_test.drop(columns=['Grade class 1: 90+  0:90-'])
    df_prediction = pandas.read_csv('predictions.csv', delimiter=';')
    df_kfold = pandas.read_csv('k_fold.csv', delimiter=';')

    for i in range(0,5):
        df_split = numpy.array_split(df_training, 5)
        df_validate = df_split.pop(i)
        df_train = pandas.concat(df_split)
        KFoldCrossValidation(df_validate, df_train, training_classes, i, df_kfold)
        df_kfold.to_csv('result_fold.csv')

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
            prediction = majority(neighbours, training_classes)
            df_prediction['Actual Grade'][index] = training_classes[index]
            df_prediction['Prediction k=' + str(k)][index] = prediction
            df_prediction['scikit k=' + str(k)][index] = neigh.predict(row.drop(['Unnamed: 0']).values.reshape(1, -1))
    df_prediction.to_csv('result.csv')



if __name__ == "__main__":
    main()
