import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import logging

logging.warning("Converting marriage and education fields into categorical........")
e1=pd.get_dummies(X['EDUCATION'],prefix='EDU')
m1=pd.get_dummies(X['MARRIAGE'],prefix='MAR')
X.drop(['EDUCATION','MARRIAGE'],axis=1,inplace=True)
X=pd.concat([X,e1,m1],axis=1)

logging.warning("Check distribution of classes.......")
df['default.payment.next.month'].hist()
plt.show()

from imblearn.over_sampling import SMOTE
logging.warning("Performing class balancing.........")
smote = SMOTE()
# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(X, y)
