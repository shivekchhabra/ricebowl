Metrics
========
Documentation of ricebowl metrics.
To use this simply do from ricebowl import metrics and then use each function with metrics.<function>

Please note all these are basic metrics outputted in a string format for your convenience.


classifier_outputs
^^^^^^^^^^^^^^^^^^
General function for producing classification metric outputs.

Parameters- y_test(expected output), y_pred(observed output), f1_average(average parameter for calculating f1 score. Optional; Default= 'micro')
Please note these parameters can be in the form of a list/ numpy array/ pandas series.


Output- Single string object containing accuracy, f1, confusion matrix and a classification report.

Usage::
    
    output = classifier_outputs(y_test, y_pred, f1_average='micro')
    print(output)


regression_outputs
^^^^^^^^^^^^^^^^^^
General function for producing regression metric outputs.

Parameters- y_test(expected output), y_pred(observed output)
Please note these parameters can be in the form of a list/ numpy array/ pandas series.


Output- Single string object containing r2 score, mape and rmse.

Usage::

    output = regression_outputs(y_test, y_pred)
    print(output)



