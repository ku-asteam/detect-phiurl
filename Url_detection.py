from __future__ import print_function
import sys
import os
import subprocess
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss, AllKNN,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection, TomekLinks,
                                     InstanceHardnessThreshold)
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn import metrics
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline
from sklearn.externals import joblib
# joblib.dump(grid.best_estimator_, "filename.pkl")
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def measure_performance(X, y, clf, show_accuracy=True,
                        show_classification_report=True,
                        show_confusion_matrix=True):
  y_pred = clf.predict(X)

  if show_accuracy:
    print("Accuracy: {0:.3f}".format(metrics.accuracy_score(y, y_pred)), "\n")

  if show_classification_report:
    print("Classification report")
    print(metrics.classification_report(y, y_pred), "\n")

  if show_confusion_matrix:
    print("Confusion matrix")
    print(metrics.confusion_matrix(y, y_pred), "\n")

def get_data(filename):
  if os.path.exists(filename):
    print("-- csv file found locally")
    dtrain = pd.read_csv(filename, encoding="utf-8-sig")
  else:
    print("-- no such file [train]")

  return dtrain

def main():
  df = get_data("Features_new_data.csv")

  clf_name = sys.argv[2]		# ["rfc", "ada", "svm", "sgd"]
  sampler_id = int(sys.argv[1])	# integer 0 ~ 17.  -1 for no-sampler option
  score = "f1"					# scoring method (optional)

  if len(sys.argv) > 3:
    score = sys.argv[3]

  # Build Classifier ---------------------------------------------------------
  param_grid = {}
  clf = ""

  if clf_name == "rfc":
    param_grid = {
                    "criterion": ["gini", "entropy"],
                    "min_samples_split": [2, 4, 6, 8, 10],
                    "bootstrap": [True, False],
                    "n_estimators": [100],
                    "max_features": ["auto"]
                 }

    # mini
    # param_grid = {
    #                 "criterion": ["gini"],
    #                 "min_samples_split": [2, 4],
    #              }

    clf = RandomForestClassifier()

  elif clf_name == "ada":
    param_grid = {
                    "base_estimator__criterion": ["gini", "entropy"],
                    "base_estimator__splitter": ["best", "random"],
                    "base_estimator__min_samples_leaf": [1, 2, 4, 6, 8, 10],
                    "n_estimators": [100]
                 }

    # mini
    # param_grid = {
    #                 "base_estimator__criterion": ["gini",],
    #                 "base_estimator__min_samples_leaf": [1, 2],
    #              }

    DTC = DecisionTreeClassifier(random_state=11,
                                 max_features="auto",
                                 class_weight="balanced",
                                 max_depth=None)

    clf = AdaBoostClassifier(base_estimator=DTC)

  elif clf_name == "svm":
    param_grid = [{"C": [1, 10, 100, 1000], "kernel": ["linear"]},
                    {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.0001], \
                                              "kernel": ["rbf"]},
                    {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.0001], \
                                              "kernel": ["sigmoid"]}]

    # mini
    # param_grid = [
    #                 {"C": [1, 10], "kernel": ["linear"]},
    #              ]

    clf = SVC()

  elif clf_name == "sgd":
    param_grid = {
                    "loss": ["modified_huber", "squared_hinge", "perceptron"],
                    "max_iter": [4000, 6000, 10000]
                 }

    # Test code Separated
    clf = SGDClassifier(class_weight="balanced")

  else:
    print("Can not find the entered classifier")

  # --------------------------------------------------------------------------
  label_col = 18
  N_FOLDS = 5

  # Run grid search ----------------------------------------------------------
  grid_search_ABC = GridSearchCV(clf, param_grid=param_grid, scoring=score)

  if sampler_id == -1: # no sampler
    clfs = []

    # build folds
    kf = KFold(n_splits=N_FOLDS, shuffle=True)
    for kf_train, kf_test in kf.split(df):
      print(kf_train)
      print(kf_test)

      result_train = df.iloc[kf_train]
      result_test = df.iloc[kf_test]

      xTrain = result_train.iloc[:, :label_col]	# data to train
      yTrain = result_train.iloc[:, label_col]	# label for xTrain

      xTest = result_test.iloc[:, :label_col]
      yTest = result_test.iloc[:, label_col]

      grid_search_ABC.fit(xTrain, yTrain)
      predictedY = grid_search_ABC.predict(xTest)

      acc_score = accuracy_score(yTest, predictedY, normalize=False)
      clf_report_imb = classification_report_imbalanced(yTest, grid_search_ABC.predict(xTest))
      print(acc_score)
      print(clf_report_imb)

      clfs.append({"accuracy_score": acc_score, "clf_report_imb": clf_report_imb})

    # Save as .p
    pkl_filename = "_".join(sys.argv) + ".p"
    with open(pkl_filename, "wb") as file:
      pickle.dump(clfs, file)

  else: # use sampler
    # Samplers list
    cc = ClusterCentroids(random_state=0, voting="auto") 
    ru = RandomUnderSampler(random_state=0)
    nm1 = NearMiss(version=1, random_state=0)
    nm2 = NearMiss(version=2, random_state=0)
    nm3 = NearMiss(version=3, random_state=0)
    enn = EditedNearestNeighbours(random_state=0)

    renn = RepeatedEditedNearestNeighbours(random_state=0)
    alknn = AllKNN(random_state=0, allow_minority=True)
    cnn = CondensedNearestNeighbour(random_state=0) 
    oss = OneSidedSelection(random_state=0)
    nc = NeighbourhoodCleaningRule(random_state=0)
    tl = TomekLinks(random_state=0)
    ih = InstanceHardnessThreshold(estimator=LogisticRegression(penalty="l1",
                                                                class_weight="balanced",
                                                                solver="liblinear"), 
                                   random_state=0)

    # oversampler
    ada = ADASYN(ratio="auto", random_state=0, n_neighbors=5, n_jobs=1)
    smt = SMOTE(ratio="auto", random_state=0, k_neighbors=5,
                m_neighbors={5, 10, 15, 20}, out_step=0.5, 
                svm_estimator=None, n_jobs=1)
    rov = RandomOverSampler(ratio="auto", random_state=0)

    # combined sampler
    smtc = SMOTEENN(random_state=0)
    smttmek = SMOTETomek(random_state=0)

    sampler_list = [cc, ru, nm1, nm2, nm3, enn, renn, alknn, cnn, oss, nc, \
                    tl, ih, ada, smt, rov, smtc, smttmek]
    x = sampler_list[sampler_id]

    clfs = []

    # build folds
    kf = KFold(n_splits=N_FOLDS, shuffle=True)
    for kf_train, kf_test in kf.split(df):
      print(kf_train)
      print(kf_test)

      result_train = df.iloc[kf_train]
      result_test = df.iloc[kf_test]

      xTrain = result_train.iloc[:, :label_col]	# data to train
      yTrain = result_train.iloc[:, label_col]	# label for xTrain

      xTest = result_test.iloc[:, :label_col]
      yTest = result_test.iloc[:, label_col]

      pipeline = make_pipeline(x, grid_search_ABC)
      pipeline.fit(xTrain, yTrain)
      predictedY = pipeline.predict(xTest)

      acc_score = accuracy_score(yTest, predictedY, normalize=False)
      clf_report_imb = classification_report_imbalanced(yTest, pipeline.predict(xTest))
      print(acc_score)
      print(clf_report_imb)

      clfs.append({"accuracy_score": acc_score, "clf_report_imb": clf_report_imb})

    # Save as .p
    pkl_filename = "_".join(sys.argv) + ".p"
    with open(pkl_filename, "wb") as file:
      pickle.dump(clfs, file)

if __name__ == "__main__":
  main()
