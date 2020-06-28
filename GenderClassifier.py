from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2


X,y = make_hastie_10_2(random_state=0)

X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0
                                 , max_depth=1, random_state=0)
pre = clf.fit(X_train, y_train)
print pre.score(X_test,y_test)

print clf.predict(X_test)
# print GradientBoostingClassifier(n_estimators=100, learning_rate=1.0
#                                  , max_depth=1, random_state=0).predict(X_test)


