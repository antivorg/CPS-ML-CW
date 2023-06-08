
# This scripts contains the functionality to run the simulation discussed in the report. Run
# this module with a command line argument defined below.
#
# python3 -m model [ARG]
#
# ARG := {help|predict|kfold|svm|randforest|xgb}
# - help: view how to run the script
# - kfold: Run k-fold cross all 3 models
# - svm: Run the parameter sweep for SVM
# - randforest: Run the parameter sweep for Rand Forest
# - xgb: Run the parameter sweep for XGBoost
#
# Author: Anthony Gaylard
# Email: aigu17@soton.ac.uk
#
#####################################################################################################


# Scripting Libraries
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

# Evaluation
from sklearn.model_selection import GridSearchCV


def print_help():

    helpString = "\nThis scripts contains the functionality to run the simulation " +\
                    "discussed in the report. Run this module with a command line " +\
                    "argument defined below.\n\npython3 -m model [ARG]\n\nARG := " +\
                    "{help|predict|kfold|svm|randforest|xgb}\n" +\
                    "- help: view how to run the script\n" +\
                    "- kfold: Run k-fold cross all 3 models\n" +\
                    "- svm: Run the parameter sweep for SVM\n" +\
                    "- randforest: Run the parameter sweep for Rand Forest\n" +\
                    "- xgb: Run the parameter sweep for XGBoost\n"

    print(helpString)


def kfold():

    # Perform the k-fold cross-validation for all three models, using parameters
    # discovered in the optimisation exercise.

    # Number of folds used
    k = 10

    dataTrain = np.genfromtxt("TrainingDataBinary.csv", delimiter=",")
    dataEval = np.genfromtxt("TestingDataBinary.csv", delimiter=",")

    folds = np.vsplit(dataTrain, k)

    
    matriciesRandForest = []
    matriciesXGB = []
    matriciesSVM = []

    # I know there is library functionality to perform the cross validation,
    # but since it is very simple to implement, I prefer to have control of
    # what's going on
    for i in range(0, k):

        trainFold = np.concatenate(np.delete(folds, i, 0))
        testFold = folds[i]

        ### Train

        X = np.delete(trainFold, 128, 1)
        y = np.delete(trainFold, np.s_[0:128], 1).flatten()

        # Create each of the models
        clf_RandForest = RandomForestClassifier(n_estimators=500, bootstrap=True)
        clf_RandForest.fit(X, y)

        clf_xgb = XGBClassifier(subsample=1, learning_rate=0.3)
        clf_xgb.fit(X, y)

        clf_svm = SVC(kernel="linear", C=5)
        clf_svm.fit(X, y)

        ### Test

        X = np.delete(testFold, 128, 1)
        y = np.delete(testFold, np.s_[0:128], 1).flatten()

        yRandForest = clf_RandForest.predict(X)
        yXGB = clf_xgb.predict(X)
        ySVM = clf_svm.predict(X)

        confusionRandForest = evaluate(y, yRandForest)
        confusionXGB = evaluate(y, yXGB)
        confusionSVM = evaluate(y, ySVM)

        print("Confusion Matrix Random Forest: " + str(confusionRandForest))
        print("Confusion Matrix XGBoost: " + str(confusionXGB))
        print("Confusion Matrix SVM: " + str(confusionSVM))

        matriciesRandForest.append(["Random Forest", confusionRandForest["accuracy"], \
                                    confusionRandForest["precision"], confusionRandForest["recall"]])
        matriciesXGB.append(["XGBoost", confusionXGB["accuracy"], \
                                    confusionXGB["precision"], confusionXGB["recall"]])
        matriciesSVM.append(["Support Vector Machines", confusionSVM["accuracy"], \
                                    confusionSVM["precision"], confusionSVM["recall"]])

    ### Evaluate

    matricies = matriciesRandForest + matriciesXGB + matriciesSVM

    # Write results to .csv
    with open("Cross-Validation.csv", 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerows(matricies)


def evaluate(y, yModel):

    # Compute the confusion matrix and other relavant parameters by comparing predicted and
    # actual labels

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


def optimise_random_forest():
    
    # This script runs a parameter sweep of Random Forest using the testing data
    # to evaluate their effect on accuracy
    #
    # Parameters optimised:
    # - n_estimator: Number of trees in the forest
    # - boostrap: Boolean to determine whether to bootstrap
    #               or use the whole data set

    parameters = {
            "estimators":[500, 1000, 1500, 2000],
            "bootstrap":[True, False]
    }

    # Split the data set into k folds and choose a random fold i for test,
    # the remaining folds are used for training.
    k = 10
    i = random.randint(0, k-1)
    dataTrain = np.genfromtxt("TrainingDataBinary.csv", delimiter=",")
    folds = np.vsplit(dataTrain, k)
    trainFold = np.concatenate(np.delete(folds, i, 0))
    testFold = folds[i]

    accuracies = []

    for estimatorsIt in parameters["estimators"]:
        for bootstrapIt in parameters["bootstrap"]:

            deltaTime = -time.time()

            print("Numberr of Estimators: " + str(estimatorsIt) + ", Bootsrap dataset(?): " + str(bootstrapIt))
            
            # Collect data for training
            X = np.delete(trainFold, 128, 1)
            y = np.delete(trainFold, np.s_[0:128], 1).flatten()

            # Create the model
            clf = RandomForestClassifier(n_estimators=estimatorsIt, bootstrap=bootstrapIt)
 
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

            deltaTime += time.time()
            print("Time: " + str(deltaTime))

            accuracies.append([estimatorsIt, bootstrapIt, confusionMat["accuracy"], deltaTime])

    # Write results to .csv
    with open("Random-Forest-Param-Sweep.csv", 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerows(accuracies)


def optimise_svm():

    # This script runs a parameter sweep of SVM using the testing data
    # to evaluate their effect on accuracy
    #
    # Parameters optimised:
    # - Kernel: hyperplane of decision bounadry
    # - C: penalty of error term for the decision boundary

    parameters = {
        'kernel':('linear', 'rbf', 'poly'),
        'C':[0.1, 1, 10, 100, 1000]
    }

    # Split the data set into k folds and choose a random fold i for test
    k = 10
    i = random.randint(0, k-1)
    dataTrain = np.genfromtxt("TrainingDataBinary.csv", delimiter=",")
    folds = np.vsplit(dataTrain, k)
    trainFold = np.concatenate(np.delete(folds, i, 0))
    testFold = folds[i]

    accuracies = []

    for kernelIt in parameters["kernel"]:
        for CIt in parameters["C"]:

            deltaTime = - time.time()

            print("Kernel: " + kernelIt + ", C: " + str(CIt))

            # Collect training data
            X = np.delete(trainFold, 128, 1)
            y = np.delete(trainFold, np.s_[0:128], 1).flatten()

            # Create the model
            clf = SVC(kernel=kernelIt, C=CIt)

            # Train it
            clf.fit(X, y)

            # Collect testing data
            X = np.delete(testFold, 128, 1)
            y = np.delete(testFold, np.s_[0:128], 1).flatten()
    
            # Predict the testing labels
            yClf = clf.predict(X)

            # Evaluate the results
            confusionMat = evaluate(y, yClf)
            print("Confusion Matrix: " + str(confusionMat))

            deltaTime += time.time()
            print("Time: " + str(deltaTime))

            accuracies.append([kernelIt, CIt, confusionMat["accuracy"], deltaTime])

    with open("SVM-Param-Sweep.csv", 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerows(accuracies)


def optimise_xgb():

    # This script runs a parameter sweep of XGBoost using the testing data
    # to evaluate their effect on accuracy
    #
    # Parameters optimised:
    # - learning_rate: Modifies the step size of model updates
    # - sub-sample: Ratio of data that is sampled prior to growing trees

    parameters = {
            "learning rate":[0.001, 0.01, 0.1, 0.5, 0.9],
            "sub-sample":[0.2, 0.5, 0.7, 0.9, 1]
    }

    # Split the data set into k folds and choose a random fold i for test
    k = 10
    i = random.randint(0, k-1)
    dataTrain = np.genfromtxt("TrainingDataBinary.csv", delimiter=",")
    folds = np.vsplit(dataTrain, k)
    trainFold = np.concatenate(np.delete(folds, i, 0))
    testFold = folds[i]

    accuracies = []

    for subSampIt in parameters["sub-sample"]:
        for learnRateIt in parameters["learning rate"]:

            deltaTime = -time.time()

            print("Sub-sample: " + str(subSampIt) + \
                    ", Learning Rate: " + str(learnRateIt))

            # Collect training data
            X = np.delete(trainFold, 128, 1)
            y = np.delete(trainFold, np.s_[0:128], 1).flatten()

            # Create the model
            clf = XGBClassifier(subsample=subSampIt, learning_rate=learnRateIt)

            # Train the model
            clf.fit(X, y)

            # Collect the testing data
            X = np.delete(testFold, 128, 1)
            y = np.delete(testFold, np.s_[0:128], 1).flatten()

            # Predict labels for testing data
            yClf = clf.predict(X)

            # Evaluate the predicition
            confusionMat = evaluate(y, yClf)
            print("Confusion Matrix: " + str(confusionMat))

            deltaTime += time.time()
            print("Time: " + str(deltaTime))

            accuracies.append([subSampIt, learnRateIt, confusionMat["accuracy"], deltaTime])

    with open("XGB-Param-Sweep.csv", 'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerows(accuracies)


def predict():

    # Using XGBoost to predict the labels for the test data,
    # a 90/10 split is used, based on the most performant fold from
    # the cross-validation process

    # Split learning and testing data
    k = 10
    # Most performent fold
    i = 3
    dataTrain = np.genfromtxt("TrainingDataBinary.csv", delimiter=",")
    folds = np.vsplit(dataTrain, k)
    trainFold = np.concatenate(np.delete(folds, i, 0))
    testFold = folds[i]

    # Collect data for training
    X = np.delete(trainFold, 128, 1)
    y = np.delete(trainFold, np.s_[0:128], 1).flatten()

    # Create the model
    clf = XGBClassifier(subsample=1, learning_rate=0.3)

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
    dataEval = np.genfromtxt("TestingDataBinary.csv", delimiter=",")

    # Evaluate it
    labels = clf.predict(dataEval)

    # Print labels to stdout
    print(labels)

    # Create TestingResultsBinary.csv (or die trying)
    results = np.column_stack((dataEval, labels))
    print(results)
    with open("TestingResultsBinary.csv","w+") as file:
        csvWriter = csv.writer(file,delimiter=',')
        csvWriter.writerows(results)



if __name__ == "__main__":

    # Entry point

    # Parse CMD args, refer to the file header
    if len(sys.argv) == 1:
        print_help()
    elif sys.argv[1] == "help":
        print_help()
    elif sys.argv[1] == "predict":
        predict()
    elif sys.argv[1] == "kfold":
        kfold()
    elif sys.argv[1] == "randforest":
        optimise_random_forest()
    elif sys.argv[1] == "svm":
        optimise_svm()
    elif sys.argv[1] == "xgb":
        optimise_xgb()
    else:
        print_help()
