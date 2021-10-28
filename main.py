import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='logs/model_development.txt',
					filemode='a',
					format='%(asctime)s %(message)s',
					datefmt="%Y-%m-%d %H:%M:%S")
logging.warning("----------")
logging.warning("MODEL CREATION STAGE.......")
logging.warning("Reading dataset..........")
df=pd.read_csv('/content/drive/MyDrive/UCI_Credit_Card.csv') ## give the link of the dataset
df.tail()

logging.warning("Extract feature set and label.........")
X=df.iloc[:,1:-1]
y=df.iloc[:,-1]
X.shape

## display the statistical characteristics of the dataset
logging.warning("get statistical characteristics.......")
df.describe()
