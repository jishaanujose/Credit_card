import numpy as np 
import pandas as pd 
from flask_ngrok import run_with_ngrok
import pickle
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
run_with_ngrok(app)
model = pickle.load(open('model.pkl','rb'))
print('model loaded')

def feature_transform(X):
    ## converting marriage and education to categorical
    ff={'ED':[0,1,2,3,4,5,6]}
    de=pd.DataFrame(ff)
    de=pd.get_dummies(ff['ED'],prefix='EDU')
    ff={'MR':[0,1,2,3]}
    de1=pd.DataFrame(ff)
    de1=pd.get_dummies(ff['MR'],prefix='MAR')
    inp=X.loc[:,'EDUCATION'].values
    inpp=inp[0]-1
    out=de.iloc[[inpp]].values
    ou=pd.DataFrame(out,columns=de.columns)
    inp1=X.loc[:,'MARRIAGE'].values
    inpp1=inp1[0]-1
    out1=de1.iloc[[inpp1]].values
    ou1=pd.DataFrame(out1,columns=de1.columns)
    X.drop(['EDUCATION','MARRIAGE'],axis=1,inplace=True)
    X=pd.concat([X,ou,ou1],axis=1)
    return X


@app.route('/')
def text():
  return render_template('text3.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]   
    name=['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    final_features = [np.array(int_features)]
    df=pd.DataFrame(final_features, columns=name)
    df1=feature_transform(df)
    final_features = df1.iloc[:,:]
    print(final_features)
    prediction = model.predict(final_features)
    if prediction==0:
      p='Not pay'
    else:
      p='Pay'

    

    return render_template('text3.html', prediction_text='Credit card prediction $ {}'.format(p))

app.run()
