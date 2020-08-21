import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, \
    classification_report


def classifier_outputs(y_test, y_pred, f1_average='micro'):
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    f1 = round(f1_score(y_test, y_pred, average=f1_average), 3)
    conf_mat = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    final = f'''The accuracy is: {accuracy}.\nThe f1-score on average = {f1_average} is: {f1}.\n
The confusion matrix:\n{conf_mat}\n
Full classification report:\n{report}'''
    return final


def regression_outputs(y_test, y_pred):
    mape = round(mean_squared_error(y_test, y_pred), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
    r2 = round(r2_score(y_test, y_pred), 3)
    final = f'''The mape is: {mape}.\nThe rmse is: {rmse}.\nThe r2 score is: {r2}\n'''
    return final
