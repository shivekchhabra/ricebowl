Modeling
========
Documentation of ricebowl modeling. 
To use this simply do from ricebowl.modeling import choose_model and then use each function with choose_model.<function> 

Please note all these are basic ML models and are set to be used with default parameters. This is solely done to achieve a base model result in shorter time for a variety of different models.

Classification Models:
======================
These are the available classification models and their function names. Please make sure to do any preprocessing beforehand using processing module from ricebowl.

random_forest_classifier
^^^^^^^^^^^^^^^^^^^^^^^^
General function for random forest classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = random_forest_classifier(data, label, test)


decision_tree_classifier
^^^^^^^^^^^^^^^^^^^^^^^^
General function for decision tree classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = decision_tree_classifier(data, label, test)


svm_classifier
^^^^^^^^^^^^^^
General function for support vector machine classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = svm_classifier(data, label, test)


extra_tree_classifier
^^^^^^^^^^^^^^^^^^^^^
General function for extra-tree classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = extra_tree_classifier(data, label, test)

 
gaussian_classifier
^^^^^^^^^^^^^^^^^^^
General function for gaussian classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = gaussian_classifier(data, label, test)


logistic_classifier
^^^^^^^^^^^^^^^^^^^
General function for logistic-regression i.e. classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = logistic_classifier(data, label, test)


logistic_cv_classifier
^^^^^^^^^^^^^^^^^^^^^^
General function for logistic regression with cross validation i.e. classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = logistic_cv_classifier(data, label, test) 


bernoulli_classifier
^^^^^^^^^^^^^^^^^^^^
General function for bernoulli classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = bernoulli_classifier(data, label, test)


multinomial_classifier
^^^^^^^^^^^^^^^^^^^^^^
General function for multinomial classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = multinomial_classifier(data, label, test)


sgd_classifier
^^^^^^^^^^^^^^
General function for stochastic gradient descent classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = sgd_classifier(data, label, test)

 
passive_aggressive_classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
General function for passive-aggressive classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = passive_aggressive_classifier(data, label, test)


ridge_classifier
^^^^^^^^^^^^^^^^
General function for ridge classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = ridge_classifier(data, label, test)


mlp_classifier
^^^^^^^^^^^^^^
General function for multi-layer-perceptron classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = mlp_classifier(data, label, test)


adaboost_classifier
^^^^^^^^^^^^^^^^^^^
General function for adaboost classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = adaboost_classifier(data, label, test)


bagging_classifier
^^^^^^^^^^^^^^^^^^
General function for bagging classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = bagging_classifier(data, label, test)


xgboost_classifier
^^^^^^^^^^^^^^^^^^
General function for xgboost classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = xgboost_classifier(data, label, test)


light_gbm_classifier
^^^^^^^^^^^^^^^^^^^^
General function for light-GBM classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = light_gbm_classifier(data, label, test)


lda_classifier
^^^^^^^^^^^^^^
General function for linear-discriminant-analysis classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = lda_classifier(data, label, test)


qda_classifier
^^^^^^^^^^^^^^
General function for quadratic-discriminant-analysis classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = qda_classifier(data, label, test)


knn_classifier
^^^^^^^^^^^^^^
General function for k-nearest-neighbour classification.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = knn_classifier(data, label, test)


Regression Models:
======================
These are the available regression models and their function names. Please make sure to do any preprocessing beforehand using processing module from ricebowl.


knn_regressor
^^^^^^^^^^^^^
General function for k-nearest-neighbour regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = knn_regressor(data, label, test)


light_gbm_regressor
^^^^^^^^^^^^^^^^^^^
General function for light-GBM regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = light_gbm_regressor(data, label, test)


xgboost_regressor
^^^^^^^^^^^^^^^^^
General function for xgboost regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = xgboost_regressor(data, label, test)


linear_regressor
^^^^^^^^^^^^^^^^
General function for linear regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = linear_regressor(data, label, test)


ransac_regressor
^^^^^^^^^^^^^^^^
General function for ransac regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = ransac_regressor(data, label, test)


ARD_regressor
^^^^^^^^^^^^^
General function for ARD regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = ARD_regressor(data, label, test)


huber_regressor
^^^^^^^^^^^^^^^
General function for huber regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = huber_regressor(data, label, test)


sgd_regressor
^^^^^^^^^^^^^
General function for stochastic-gradient-descent regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = sgd_regressor(data, label, test)


theilsen_regressor
^^^^^^^^^^^^^^^^^^
General function for theilsen regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = theilsen_regressor(data, label, test)


passive_aggressive_regressor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
General function for passive aggressive regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = passive_aggressive_regressor(data, label, test)


mlp_regressor
^^^^^^^^^^^^^
General function for multi-layered-perceptron regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = mlp_regressor(data, label, test)


adaboost_regressor
^^^^^^^^^^^^^^^^^^
General function for adaboost regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = adaboost_regressor(data, label, test)


random_forest_regressor
^^^^^^^^^^^^^^^^^^^^^^^
General function for random-forest regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = random_forest_regressor(data, label, test)


decision_tree_regressor
^^^^^^^^^^^^^^^^^^^^^^^
General function for decision tree regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = decision_tree_regressor(data, label, test)


svm_regressor
^^^^^^^^^^^^^
General function for svm regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = svm_regressor(data, label, test)


bagging_regressor
^^^^^^^^^^^^^^^^^
General function for bagging regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = bagging_regressor(data, label, test)


extra_tree_regressor
^^^^^^^^^^^^^^^^^^^^
General function for extra tree regression.

Parameters- data, label, test
Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = extra_tree_regressor(data, label, test)


