import numpy as np
import pandas as pd
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier, \
    LinearRegression, RANSACRegressor, ARDRegression, HuberRegressor, LogisticRegression, \
    LogisticRegressionCV, SGDRegressor, TheilSenRegressor, PassiveAggressiveRegressor, Lasso, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, \
    RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn import svm
# from xgboost import XGBClassifier, XGBRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, DecisionTreeRegressor, \
    ExtraTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(3)


# Specific Function to predict the output
def predicting(model, data, labels, test_data):
    final_model = model
    y_pred = final_model.fit(data, labels).predict(test_data)
    output = pd.DataFrame()
    output['predictions'] = y_pred
    return output


# ------------------------------------------CLASSIFICATION---------------------------------------------------

# General function for random forest classification
def random_forest_classifier(data, label, test):
    model = RandomForestClassifier()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for decision tree classification
def decision_tree_classifier(data, label, test):
    model = DecisionTreeClassifier()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for support vector machine classification
def svm_classifier(data, label, test):
    model = svm.SVC()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for extra-tree classification
def extra_tree_classifier(data, label, test):
    model = ExtraTreeClassifier()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for gaussian classification
def gaussian_classifier(data, label, test):
    model = GaussianNB()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for logistic-regression i.e. classification
def logistic_classifier(data, label, test):
    model = LogisticRegression()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for logistic-regression with cross validation i.e. classification
def logistic_cv_classifier(data, label, test):
    model = LogisticRegressionCV()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for bernoulli classification
def bernoulli_classifier(data, label, test):
    model = BernoulliNB()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for multinomial classification
def multinomial_classifier(data, label, test):
    model = MultinomialNB()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for stochastic gradient descent classification
def sgd_classifier(data, label, test):
    model = SGDClassifier()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for passive aggressive classification
def passive_aggressive_classifier(data, label, test):
    model = PassiveAggressiveClassifier()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for ridge classification
def ridge_classifier(data, label, test):
    model = RidgeClassifier()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for multi-layer-perceptron classification
def mlp_classifier(data, label, test):
    model = MLPClassifier()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for adaboost classification
def adaboost_classifier(data, label, test):
    model = AdaBoostClassifier()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for bagging classification
def bagging_classifier(data, label, test):
    model = BaggingClassifier()
    ypred = predicting(model, data, label, test)
    return ypred


# # General function for xgboost classification
# def xgboost_classifier(data, label, test):
#     model = XGBClassifier()
#     ypred = predicting(model, data, label, test)
#     return ypred


# # General function for light-GBM classification
# def light_gbm_classifier(data, label, test):
#     model = LGBMClassifier()
#     ypred = predicting(model, data, label, test)
#     return ypred


# General function for linear-discriminant-analysis classification
def lda_classifier(data, label, test):
    model = LinearDiscriminantAnalysis()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for quadratic-discriminant-analysis classification
def qda_classifier(data, label, test):
    model = QuadraticDiscriminantAnalysis()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for k-nearest-neighbour classification
def knn_classifier(data, label, test):
    model = KNeighborsClassifier()
    ypred = predicting(model, data, label, test)
    return ypred


# ------------------------------------------REGRESSION---------------------------------------------------


# General function for k-nearest-neighbour regression
def knn_regressor(data, label, test):
    model = KNeighborsRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# # General function for light-GBM regression
# def light_gbm_regressor(data, label, test):
#     model = LGBMRegressor()
#     ypred = predicting(model, data, label, test)
#     return ypred


# # General function for xgboost regression
# def xgboost_regressor(data, label, test):
#     model = XGBRegressor()
#     ypred = predicting(model, data, label, test)
#     return ypred


# General function for linear regression
def linear_regressor(data, label, test):
    model = LinearRegression()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for ransac regression
def ransac_regressor(data, label, test):
    model = RANSACRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for ARD regression
def ARD_regressor(data, label, test):
    model = ARDRegression()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for huber regression
def huber_regressor(data, label, test):
    model = HuberRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for stochastic-gradient-descent regression
def sgd_regressor(data, label, test):
    model = SGDRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for theilsen regression
def theilsen_regressor(data, label, test):
    model = TheilSenRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for passive aggressive regression
def passive_aggressive_regressor(data, label, test):
    model = PassiveAggressiveRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for multi-layered-perceptron regression
def mlp_regressor(data, label, test):
    model = MLPRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for adaboost regression
def adaboost_regressor(data, label, test):
    model = AdaBoostRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for random_forest regression
def random_forest_regressor(data, label, test):
    model = RandomForestRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for decision-treee regression
def decision_tree_regressor(data, label, test):
    model = DecisionTreeRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for support vector machine regression
def svm_regressor(data, label, test):
    model = svm.SVR()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for bagging regression
def bagging_regressor(data, label, test):
    model = BaggingRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for extra-tree regression
def extra_tree_regressor(data, label, test):
    model = ExtraTreeRegressor()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for Lasso regression
def lasso_regressor(data, label, test):
    model = Lasso()
    ypred = predicting(model, data, label, test)
    return ypred


# General function for Ridge regression
def ridge_regressor(data, label, test):
    model = Ridge()
    ypred = predicting(model, data, label, test)
    return ypred
