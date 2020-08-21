import numpy as np

np.random.seed(7)


def predict(model, data, test_data, labels):
    final_model = model
    y_pred = final_model.fit(data, labels).predict(test_data)
    return y_pred
