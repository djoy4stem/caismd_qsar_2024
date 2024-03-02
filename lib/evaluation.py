import numpy as np
import pandas as pd
from typing import Union



def evaluate_binary_classifier(model, X:pd.DataFrame, y:Union[pd.DataFrame, pd.Series, np.array], metrics:list
                        , proba_threshold:float=None):
    y_test_pred = None

    if proba_threshold is None:
        y_test_pred = model.predict(X)
    else:
        # print("predict probabilities")
        y_test_proba = model.predict_proba(X)
        # y_test_pred  = np.argmax(y_test_proba, axis=1)
        y_test_pred = [int(x[1]>=proba_threshold) for x in y_test_proba]
        
    return calculate_binary_class_scores(y_test_pred, y, metrics, proba_threshold=proba_threshold)


def calculate_binary_class_scores(y_true, y_pred, metrics, proba_threshold=None):
    y_pred_ = None
    if (isinstance(y_pred, np.ndarray) and y_pred.ndim==2) or (isinstance(y_pred, (np.ndarray, list)) and isinstance(y_pred[0], list) and len(y_pred[0]==2)):
        if not proba_threshold is None:
            # print("from predict_proba")
            y_pred_ = [int(x[1]>=proba_threshold) for x in y_pred]
        else:
            raise ValueError("It seems you provided a classs probabilities. You must specify a non-null probability threshold.")
    elif (isinstance(y_pred, np.ndarray) and y_pred.ndim==1) or (isinstance(y_pred, (np.ndarray, list)) and isinstance(y_pred[0], (int, float))):
        y_pred_ = y_pred
    
    scores = {}
    # print(y_true)
    # print(y_pred)
    for metric in metrics:
        try:

            scores[metric.__name__] = round(metric(y_true, y_pred_), 3)
        except Exception as exp:
            print(f"Could not compute metric '{metric.__name__}'.\n{exp}")

    return scores