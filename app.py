from flask import Flask,render_template,request
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    df = pd.read_csv("USA_Predictors_of_How_Counties_Vote.csv") 
    x=df.drop(["Margin D"],axis=1).values
    y=df["Margin D"].values
    x_train,x_test,y_train,y_test,=train_test_split(x,y,test_size=0.2,random_state=0)
    model = MLPRegressor(random_state=1, max_iter=500).fit(x_train, y_train)
    pickle.dump(model,open("vote.pkl","wb"))
    return render_template("home.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    HO=request.form['HO']
    MC=request.form["MC"]
    PV=request.form["PV"]
    BD=request.form["BD"]
    DO=request.form["DO"]
    WT=request.form["WT"]
    BK=request.form["BK"]
    NA=request.form["NA"]
    AS=request.form["AS"]
    HS=request.form["HS"]
    FB=request.form["FB"]

    form_array=np.array([[HO,MC,PV,BD,DO,WT,BK,NA,AS,HS,FB]])
    #model=pickle.load(open("vote.pkl"),"rb")
    with open('vote.pkl','rb') as f:
        model=pickle.load(f)

    prediction=model.predict(form_array)[0]
    result=prediction
    return render_template("final.html",result=result)


if __name__ == "__main__":
    app.run(debug=True)