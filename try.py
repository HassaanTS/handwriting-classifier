"""
===> G.C.U.H v1.0  8th May 2016
Classified Gender form handwriting using Gradient boosting decision trees.
We have used already extracted features for now.
"""

from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import csv

train_answers = pd.io.parsers.read_csv("train_answers.csv" )#, header=0 ,skiprows=1  )
y = train_answers['male'].values

y = np.array(y, dtype='float_')

#
#
# #
# # f = open("train.csv", 'r')
train = pd.io.parsers.read_csv("train.csv")

# test = pd.io.parsers.read_csv("test.csv")
#
#
#
f = open("features.csv", 'r')
features = csv.reader(f)
#
#
#
feat =[]

for i in features:
    #print i
    temp = ""
    first = True

    for j in i[0]:
      if j == '.' and first is True:
          j= '['
          temp+=j
          first = False
      elif j == '.' and first is False:
          j = ']'
          temp += j
          first = True
      else:
          temp += j

    feat.append(temp)


# Keeping only the features required form train.csv
train = train[feat]
X = train.as_matrix()

X_train = X[3::4]#X-train contains 4th row traing features english
                 #y contains labels, smae lenght as X_train


print X_train.shape
print y.shape
# y = y.reshape(len(y), 1)


# print feat
#
obj = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0
                                  , max_depth=5, random_state=0)
#
pre = obj.fit(X_train, y)

#================================================TESTING=============================
test_data = pd.io.parsers.read_csv("test.csv")
test_data = test_data[feat]
X_test = test_data.as_matrix()

X_test = X_test[3::4] # filter the test data same as the training data
# print "X_test shape:",X_test.shape
# print "X_train shape:",X_train.shape
# print "y shape:",y.shape
#
predict = obj.predict(X_test)
print predict


#print obj.predict(test)




#
# i=0
# for j in train_answers:
#     print j
#     i+=1
#     if i ==11:
#         break




