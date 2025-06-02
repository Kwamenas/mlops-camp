#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import mlflow
import os
import xgboost as xgb
import mlflow.xgboost
import mlflow.sklearn
import mlflow.pyfunc
from pandas import DataFrame,Series
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import root_mean_squared_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials
from hyperopt.pyll import scope
from mlflow import MlflowClient
from mlflow.entities import ViewType



mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('pred-nyc')

os.makedirs('models', exist_ok=True)
os.makedirs('data_pre',exist_ok=True)


# # Load the Data
def read_and_preprocess(year,month):
    url=f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month}.parquet'
    df=pd.read_parquet(url)

    df['trip_duration_min']=(df['lpep_dropoff_datetime']-df['lpep_pickup_datetime']).dt.total_seconds()/60
    df=df[(df['trip_duration_min'] >=1)&(df['trip_duration_min'] <=60)]

    categorical=['PULocationID','DOLocationID']
    #numerical=['trip_distance']

    df[categorical]=df[categorical].astype('category')
    return df

#Preprocess our Data
def create_X_data(df,dv=None):
    #the categorical and numerical values
    categorical=['PULocationID','DOLocationID']
    numerical=['trip_distance']
    train_dic=df[numerical+categorical].to_dict(orient='records')
    
    if dv is None:
        dv=DictVectorizer()
        X=dv.fit_transform(train_dic)
    else:
        X=dv.transform(train_dic)
    
    return X,dv

#Split our data and train the data
#Also we log the data here
def train_model(X_train,y_train,X_val,y_val,dv):
    with mlflow.start_run() as run:
        dtrain=xgb.DMatrix(X_train,label=y_train)
        dvalid=xgb.DMatrix(X_val,label=y_val)
        
        f_params={'reg_lambda':0.007269041257774687,
        'min_child_weight':3.7097356295703907,
        'learning_rate':0.24288674966558005,
        'objective':'reg:linear',
        'seed':42,
        'reg_alpha':0.009307293737791112,
        'max_depth':9}

        mlflow.log_params(f_params)

        booster=xgb.train(
                params=f_params,
                dtrain=dtrain,
                num_boost_round=15,
                evals=[(dvalid,"validation")],
                early_stopping_rounds=10
            )
        y_pred=booster.predict(dvalid)
        rmse=mean_squared_error(y_val,y_pred)
        rmse_=np.sqrt(rmse)
        mlflow.log_metric("rmse",rmse_)
        
        with open('models/preprocessor.bin','wb') as f_out:
            pickle.dump(dv,f_out)
        mlflow.log_artifact(local_path='models/preprocessor.bin',artifact_path='preprocessor')
        mlflow.xgboost.log_model(booster,artifact_path='models_mlflow')

        return run.info.run_id

#We run our data here
def run_model(year,month):
    #logging the data using the read_and _preprocess
    df_train=read_and_preprocess(year=year,month=month)
    df_train.to_csv('data_pre/train_data.csv',index=False)

    next_year = year if month < 12 else year + 1
    next_month = month +1 if month < 12 else 1

    df_val=read_and_preprocess(year=next_year,month=next_month)
    df_val.to_csv('data_pre/train_data.csv',index=False)

    ##Create X
    X_train,dv=create_X_data(df_train)
    X_val,_=create_X_data(df_val,dv)
    
    target='trip_duration_min'
    y_train=df_train[target].values
    y_val=df_val[target].values

    #run model
    run_id=train_model(X_train,y_train,X_val,y_val,dv)
    print (f"MLflow run_id:{run_id}")   
    return run_id

