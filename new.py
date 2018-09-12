
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:24:53 2018

@author: xin

"""

from sklearn.model_selection import train_test_split
import MyMethods as myFunc

X, y = myFunc.importRawDataCleaning()

# Split the dataset in two parts : train 70%, test 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
train_class = y_train.value_counts()
test_class = y_test.value_counts()
print("class distribution for balance: train: \n"+str(train_class)+"test: \n"+str(test_class))

# standarlize training data and test data
X_train, X_test = myFunc.dataStandarlize(X_train, X_test)
y_train.index = range(len(y_train)) #reset index from 0 to match train data index
y_test.index = range(len(y_test))


# feature selection
num_class = y_train.value_counts
X_train, X_test = myFunc.featureSel(X_train, y_train, X_test, num_class)


# parameter estimation
myFunc.paramEstimateSVM(X_train, X_test, y_train, y_test)
myFunc.paramEstimateRF(X_train, y_train)
myFunc.paramEstimateKNN(X_train, y_train)

# predict test set and get AUC with cross_validation to compare various models
myFunc.TrainClassification(X_train, X_test, y_train, y_test)


