import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

logging.warning("Split data into train and test sets.......")
(X_train,X_test,y_train,y_test)=train_test_split(x_smote,y_smote,test_size=0.3,random_state=42)
print(X_train.shape)
print(X_test.shape)

logging.warning("Initializing classifier list...")
result=[]
clff=['Random Forest','Extra Tree','CatBoost']
classifiers=[RandomForestClassifier(n_estimators=150,max_depth=15,min_samples_split=2),
             ExtraTreesClassifier(n_estimators=150, max_depth=15,min_samples_split=2),
             CatBoostClassifier(random_state=42, silent=True)]
logging.warning("Classifier training and testing.....")
for clf in classifiers:
  clf.fit(X_train,y_train)
  y_pred=clf.predict(X_test)
  acc=accuracy_score(y_test,y_pred)
  result.append(acc)
logging.warning("Generating dataframe with classifier type and accuracy...")
out=pd.DataFrame(data={'classifier':clff,'accuracy':result})
print(out.head())

## identify the best classifier
logging.warning("Identifying the best classifier model.....")
se=out['accuracy'].idxmax()
clf_sel=classifiers[se]


logging.warning("Best classifier fitting.....")
clf_sel.fit(X_train,y_train)
logging.warning("Display classification report.....")
y_pred=clf_sel.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

logging.warning("Saving model...")
m1=clf_sel
pickle.dump(m1, open('model.pkl','wb'))
