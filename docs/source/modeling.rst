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

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = random_forest_classifier(training_data, training_label, test_data)


decision_tree_classifier
^^^^^^^^^^^^^^^^^^^^^^^^
General function for decision tree classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = decision_tree_classifier(training_data, training_label, test_data)


svm_classifier
^^^^^^^^^^^^^^
General function for support vector machine classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = svm_classifier(training_data, training_label, test_data)


extra_tree_classifier
^^^^^^^^^^^^^^^^^^^^^
General function for extra-tree classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = extra_tree_classifier(training_data, training_label, test_data)

 
gaussian_classifier
^^^^^^^^^^^^^^^^^^^
General function for gaussian classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = gaussian_classifier(training_data, training_label, test_data)


logistic_classifier
^^^^^^^^^^^^^^^^^^^
General function for logistic-regression i.e. classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = logistic_classifier(training_data, training_label, test_data)


logistic_cv_classifier
^^^^^^^^^^^^^^^^^^^^^^
General function for logistic regression with cross validation i.e. classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = logistic_cv_classifier(training_data, training_label, test_data) 


bernoulli_classifier
^^^^^^^^^^^^^^^^^^^^
General function for bernoulli classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = bernoulli_classifier(training_data, training_label, test_data)


multinomial_classifier
^^^^^^^^^^^^^^^^^^^^^^
General function for multinomial classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = multinomial_classifier(training_data, training_label, test_data)


sgd_classifier
^^^^^^^^^^^^^^
General function for stochastic gradient descent classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = sgd_classifier(training_data, training_label, test_data)

 
passive_aggressive_classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
General function for passive-aggressive classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = passive_aggressive_classifier(training_data, training_label, test_data)


ridge_classifier
^^^^^^^^^^^^^^^^
General function for ridge classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = ridge_classifier(training_data, training_label, test_data)


mlp_classifier
^^^^^^^^^^^^^^
General function for multi-layer-perceptron classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = mlp_classifier(training_data, training_label, test_data)


adaboost_classifier
^^^^^^^^^^^^^^^^^^^
General function for adaboost classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = adaboost_classifier(training_data, training_label, test_data)


bagging_classifier
^^^^^^^^^^^^^^^^^^
General function for bagging classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = bagging_classifier(training_data, training_label, test_data)





lda_classifier
^^^^^^^^^^^^^^
General function for linear-discriminant-analysis classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = lda_classifier(training_data, training_label, test_data)


qda_classifier
^^^^^^^^^^^^^^
General function for quadratic-discriminant-analysis classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = qda_classifier(training_data, training_label, test_data)


knn_classifier
^^^^^^^^^^^^^^
General function for k-nearest-neighbour classification.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate accuracy, f1, confusion matrix and a classification report.

Usage::
    
    ypred = knn_classifier(training_data, training_label, test_data)


Regression Models:
======================
These are the available regression models and their function names. Please make sure to do any preprocessing beforehand using processing module from ricebowl.


knn_regressor
^^^^^^^^^^^^^
General function for k-nearest-neighbour regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = knn_regressor(training_data, training_label, test_data)




linear_regressor
^^^^^^^^^^^^^^^^
General function for linear regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = linear_regressor(training_data, training_label, test_data)


ransac_regressor
^^^^^^^^^^^^^^^^
General function for ransac regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = ransac_regressor(training_data, training_label, test_data)


ARD_regressor
^^^^^^^^^^^^^
General function for ARD regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = ARD_regressor(training_data, training_label, test_data)


huber_regressor
^^^^^^^^^^^^^^^
General function for huber regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = huber_regressor(training_data, training_label, test_data)


sgd_regressor
^^^^^^^^^^^^^
General function for stochastic-gradient-descent regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = sgd_regressor(training_data, training_label, test_data)


theilsen_regressor
^^^^^^^^^^^^^^^^^^
General function for theilsen regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = theilsen_regressor(training_data, training_label, test_data)


passive_aggressive_regressor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
General function for passive aggressive regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = passive_aggressive_regressor(training_data, training_label, test_data)


mlp_regressor
^^^^^^^^^^^^^
General function for multi-layered-perceptron regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = mlp_regressor(training_data, training_label, test_data)


adaboost_regressor
^^^^^^^^^^^^^^^^^^
General function for adaboost regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = adaboost_regressor(training_data, training_label, test_data)


random_forest_regressor
^^^^^^^^^^^^^^^^^^^^^^^
General function for random-forest regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = random_forest_regressor(training_data, training_label, test_data)


decision_tree_regressor
^^^^^^^^^^^^^^^^^^^^^^^
General function for decision tree regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = decision_tree_regressor(training_data, training_label, test_data)


svm_regressor
^^^^^^^^^^^^^
General function for svm regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = svm_regressor(training_data, training_label, test_data)


bagging_regressor
^^^^^^^^^^^^^^^^^
General function for bagging regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = bagging_regressor(training_data, training_label, test_data)


extra_tree_regressor
^^^^^^^^^^^^^^^^^^^^
General function for extra tree regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = extra_tree_regressor(training_data, training_label, test_data)


lasso_regressor
^^^^^^^^^^^^^^^
General function for Lasso regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = lasso_regressor(training_data, training_label, test_data)


ridge_regressor
^^^^^^^^^^^^^^^
General function for Ridge regression.

Parameters- training data, training label, test data

Please note these parameters can be in the form of a list/ numpy array/ pandas dataframes.


Output- Predicted values in the form of a dataframe series. These can then be used as is or with metrics module to generate rmse, r2 score and mape.

Usage::

    ypred = ridge_regressor(training_data, training_label, test_data)


