{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "flask_app.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyMCgLtF4KWRhNVfB578QyZU"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "X1VE3SYixZdD",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f81a4dcd-5a9e-461d-8eee-a36bd9f8f147"
      },
      "source": [
        "import numpy as np \n",
        "import pandas as pd \n",
        "from flask_ngrok import run_with_ngrok\n",
        "import pickle\n",
        "from flask import Flask, request, jsonify, render_template\n",
        "app = Flask(__name__)\n",
        "run_with_ngrok(app)\n",
        "model = pickle.load(open('model.pkl','rb'))\n",
        "print('model loaded')\n",
        "\n",
        "def feature_transform(X):\n",
        "    ## converting marriage and education to categorical\n",
        "    ff={'ED':[0,1,2,3,4,5,6]}\n",
        "    de=pd.DataFrame(ff)\n",
        "    de=pd.get_dummies(ff['ED'],prefix='EDU')\n",
        "    ff={'MR':[0,1,2,3]}\n",
        "    de1=pd.DataFrame(ff)\n",
        "    de1=pd.get_dummies(ff['MR'],prefix='MAR')\n",
        "    inp=X.loc[:,'EDUCATION'].values\n",
        "    inpp=inp[0]-1\n",
        "    out=de.iloc[[inpp]].values\n",
        "    ou=pd.DataFrame(out,columns=de.columns)\n",
        "    inp1=X.loc[:,'MARRIAGE'].values\n",
        "    inpp1=inp1[0]-1\n",
        "    out1=de1.iloc[[inpp1]].values\n",
        "    ou1=pd.DataFrame(out1,columns=de1.columns)\n",
        "    X.drop(['EDUCATION','MARRIAGE'],axis=1,inplace=True)\n",
        "    X=pd.concat([X,ou,ou1],axis=1)\n",
        "    return X\n",
        "\n",
        "\n",
        "@app.route('/')\n",
        "def text():\n",
        "  return render_template('text3.html')\n",
        "\n",
        "@app.route('/predict',methods=['POST'])\n",
        "def predict():\n",
        "    '''\n",
        "    For rendering results on HTML GUI\n",
        "    '''\n",
        "    int_features = [int(x) for x in request.form.values()]   \n",
        "    name=['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']\n",
        "    final_features = [np.array(int_features)]\n",
        "    df=pd.DataFrame(final_features, columns=name)\n",
        "    df1=feature_transform(df)\n",
        "    final_features = df1.iloc[:,:]\n",
        "    print(final_features)\n",
        "    prediction = model.predict(final_features)\n",
        "    if prediction==0:\n",
        "      p='Not pay'\n",
        "    else:\n",
        "      p='Pay'\n",
        "\n",
        "    \n",
        "\n",
        "    return render_template('text3.html', prediction_text='Credit card prediction $ {}'.format(p))\n",
        "\n",
        "app.run()"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "model loaded\n",
            " * Running on http://fd40-35-194-160-79.ngrok.io\n",
            " * Traffic stats available on http://127.0.0.1:4040\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "127.0.0.1 - - [26/Oct/2021 02:18:35] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
            "127.0.0.1 - - [26/Oct/2021 02:18:36] \"\u001b[33mGET /favicon.ico HTTP/1.1\u001b[0m\" 404 -\n",
            "127.0.0.1 - - [26/Oct/2021 02:19:22] \"\u001b[37mPOST /predict HTTP/1.1\u001b[0m\" 200 -\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "   LIMIT_BAL  SEX  AGE  PAY_0  PAY_2  ...  EDU_6  MAR_0  MAR_1  MAR_2  MAR_3\n",
            "0      20000    2   42      4      2  ...      0      1      0      0      0\n",
            "\n",
            "[1 rows x 32 columns]\n"
          ]
        }
      ]
    }
  ]
}
