#
# This scripts contains the functionality to run the simulation discussed in the report. Run
# this module with a command line argument defined below.
#
# python3 -m model [ARG]
#
# ARG := {help|kfold|predict|shuffle}
# - help: view how to run the script
# - kfold: Run k-fold validation
# - predict: Run the inference that creates the labels\n
# - shuffle: Create a shuffled version of the training set
#
# Author: Anthony Gaylard
# Email: aig1u17@soton.ac.uk
#
#############################################################################################



import csv
import sys
import random
import time

# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Data Engineering
from sklearn import preprocessing

# Evaluation
from sklearn.model_selection import GridSearchCV


def print_help():

    helpString = "\nThis scripts contains the functionality to run the simulation " +\
                    "discussed in the report. Run this module with a command line " +\
                    "argument defined below.\n\npython3 -m model [ARG]\n\nARG := " +\
                    "{help|kfold|predict|shuffle}\n" +\
                    "- help: view how to run the script\n" +\
                    "- kfold: Run k-fold validation\n" +\
                    "- predict: Run the inference that creates the labels\n" +\
                    "- shuffle: Create a shuffled version of the training set\n"

    print(helpString)


def kfold():

    # Number of folds used
    k = 10

    dataTrain = np.genfromtxt("ShuffledTrainingDataMulti.csv", delimiter=",")

    folds = np.vsplit(dataTrain, k)

    foldMatricies = []

    # Run through the folds
    for i in range(0, k):

        trainFold = np.concatenate(np.delete(folds, i, 0))
        testFold = folds[i]

        ### Train

        X = np.delete(trainFold, 128, 1)
        y = np.delete(trainFold, np.s_[0:128], 1).flatten()

        # Create each of the models
        clf_RandForest = RandomForestClassifier(n_estimators=500, bootstrap=True)
        clf_RandForest.fit(X, y)

        ### Test

        X = np.delete(testFold, 128, 1)
        y = np.delete(testFold, np.s_[0:128], 1).flatten()

        yInfer = clf_RandForest.predict(X)

        confusion = evaluate(y, yInfer)

        print("Confusion Matrix: " + str(confusion))

        foldMatricies.append(["Fold " + str(i), confusion["accuracy"], confusion["precision"], confusion["recall"]])

    # Write results to .csv
    with open("K-Fold-Evaluation.csv", 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerows(foldMatricies)


def evaluate(y, yModel):

    # Confusion Matrix + Useful evalution parameters
    confusionMatrix = {"tp":0, "fp":0, "tn":0 , "fn":0, "accuracy":0, "precision":0, "recall":0}

    # Evaluate the confusion matrix accross the data
    for i in np.nditer(y):
        for j in np.nditer(yModel):
            if i and j:
                confusionMatrix["tp"] += 1
            if i and not j:
                confusionMatrix["fp"] += 1
            if not i and j:
                confusionMatrix["fn"] += 1
            else:
                confusionMatrix["tn"] += 1

    # Accuracy = (tp + tn) / (tp + tn + fn + fp)
    confusionMatrix["accuracy"] = (confusionMatrix["tp"] + confusionMatrix["tn"]) / \
                                    (confusionMatrix["tp"] + confusionMatrix["tn"] + \
                                    confusionMatrix["fp"]+confusionMatrix["fn"])

    # Precision = tp / (tp + fp)
    if (confusionMatrix["tp"] + confusionMatrix["fp"]) == 0:
         confusionMatrix["precision"] = "inf"
    else:
        confusionMatrix["precision"] = confusionMatrix["tp"] / \
                                        (confusionMatrix["tp"] + confusionMatrix["fp"])

    # Recall = tp / (tp + fn)
    if (confusionMatrix["tp"]+confusionMatrix["fn"]) == 0:
        confusionMatrix["recall"] = "inf"
    else:
        confusionMatrix["recall"] = confusionMatrix["tp"] / \
                                    (confusionMatrix["tp"]+confusionMatrix["fn"])

    return confusionMatrix



def predict():

    # Using Random Forest to predict the labels for the test data,
    # a 90/10 split is used, based on the most performant fold from
    # the k-fold process

    # Split learning and testing data
    k = 10
    # Most performent fold
    i = 4
    dataTrain = np.genfromtxt("ShuffledTrainingDataMulti.csv", delimiter=",")
    folds = np.vsplit(dataTrain, k)
    trainFold = np.concatenate(np.delete(folds, i, 0))
    testFold = folds[i]

    # Collect data for training
    X = np.delete(trainFold, 128, 1)
    y = np.delete(trainFold, np.s_[0:128], 1).flatten()

    # Create the model
    clf = RandomForestClassifier(n_estimators=500, bootstrap=True)

    # Train the model
    clf.fit(X, y)

    # Collect data for testing
    X = np.delete(testFold, 128, 1)
    y = np.delete(testFold, np.s_[0:128], 1).flatten()

    # Perform predicitions on the test data
    yClf = clf.predict(X)

    # Evaluate the predictions on the test data
    confusionMat = evaluate(y, yClf)
    print("Confusion Matrix: " + str(confusionMat))

    # Bring in the testing data
    dataEval = np.genfromtxt("TestingDataMulti.csv", delimiter=",")

    # Evaluate it
    labels = clf.predict(dataEval)

    # Print labels to stdout
    print(labels)

    # Create TestingResultsBinary.csv
    results = np.column_stack((dataEval, labels))
    print(results)
    with open("TestingResultsBinary.csv","w+") as file:
        csvWriter = csv.writer(file,delimiter=',')
        csvWriter.writerows(results)


def shuffle_file():

    dataTrain = np.genfromtxt("TrainingDataMulti.csv", delimiter=",")

    np.random.shuffle(dataTrain)

    np.savetxt("ShuffledTrainingDataMulti.csv", dataTrain, delimiter=",")



if __name__ == "__main__":

    # Parse CMD args
    if len(sys.argv) == 1:
        print_help()
    elif sys.argv[1] == "help":
        print_help()
    elif sys.argv[1] == "predict":
        predict()
    elif sys.argv[1] == "kfold":
        kfold()
    elif sys.argv[1] == "shuffle":
        shuffle_file()
    else:
        print_help()
