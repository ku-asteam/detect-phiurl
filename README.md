# Detect-phiurl
Phishing URL Detection Based on Strings

## Introduction
PRank Score based on the Matrix Form is a implementation of aSTEAM Project (Next-Generation Information Computing Development Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT). The function of this software is to learn the classifier with the phishing URL feature data extracted from the URL string. In addition, several under / oversampling methods are applied together to solve the class imbalance problem.

## Requirements and Dependencies
* Above development was based on the Python version of 3.5 (`64bit`)
* Please import packages (`sklearn.ensemble.RandomForestClassifier, sklearn.tree.DecisionTreeClassifier, sklearn.linear_model.LogisticRegression, SGDClassifier etc`)

## Instructions
Entering a classifier name, such as RandomForestClassifier, AdaBoostClassifier, SGDClassifier, will classifier whether url is malicious or not, and compare the classification accuracy of each classifier.

