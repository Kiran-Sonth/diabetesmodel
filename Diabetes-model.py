import pandas as pd
import numpy as np
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv(r"C:\Users\Dell-Pc\Downloads\diabetes.csv")

data["Glucose"]=data["Glucose"].replace(0,data["Glucose"].mean())
data["BloodPressure"]=data["BloodPressure"].replace(0,data["BloodPressure"].mean())
data["SkinThickness"]=data["SkinThickness"].replace(0,data["SkinThickness"].mean())
data["Insulin"]=data["Insulin"].replace(0,data["Insulin"].mean())
data["BMI"]=data["BMI"].replace(0,data["BMI"].mean())


sampler=RandomOverSampler(random_state=45)
x_res=data.iloc[:,0:8].values
y_res=data["Outcome"].values
x_res, y_res = sampler.fit_resample(x_res, y_res)


x_train,x_test,y_train,y_test=train_test_split(x_res,y_res,test_size=0.25,random_state=0)

model=RandomForestClassifier(n_estimators=50,criterion="entropy",random_state=10)
model.fit(x_train,y_train)

filename = r'C:\Users\Dell-Pc\kiran\Diabetes-prediction-model.pkl'
pickle.dump(model,open(filename,'wb'))





