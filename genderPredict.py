from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import math
def logit(pred):
    if (len())


def logistic(logit, A, B):
    return (1/ (1 + math.exp(A*logit + B)))

def calibrate(pred):
    A = -1
    B = 0

    pred = logistic(logit(pred), A, B)
    upper = 0.9
    lower = 0.1

    pred[pred>upper] = (pred[pred > upper] + upper)/2
    pred[pred<lower] = (pred[pred < lower] + lower)/2

    return pred

def model(train, target, test):
    n = GradientBoostingRegressor(n_estimators=80, learning_rate=1.0
                                 , max_depth=5, random_state=0)
    clf = n.fit(train,target)
    pred = n.predict(test)
    pred = np.asarray(pred)
    pred.reshape()
