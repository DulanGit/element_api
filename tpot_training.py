# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV 
from tpot import TPOTClassifier
import matplotlib.pyplot as plt
from scipy import signal 
import pickle

import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

dataframe_hrv = pd.read_csv("dataframe_hrv.csv")
dataframe_hrv = dataframe_hrv.reset_index(drop=True)
dataframe_hrv = dataframe_hrv.reset_index()
display(dataframe_hrv.head(5))

def fix_stress_labels(df='',label_column='stress'):
    df['stress'] = np.where(df['stress']>=0.5, 1, 0)
    display(df["stress"].unique())
    return df
dataframe_hrv = fix_stress_labels(df=dataframe_hrv)
temp = np.where(dataframe_hrv['stress']>=0.5, 1, 0)

def missing_values(df):
    df = df.reset_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df[~np.isfinite(df)] = np.nan
    df.plot( y=["HR"])
    df['HR'].fillna((df['HR'].mean()), inplace=True)
    df['HR'] = signal.medfilt(df['HR'],13) 
    df.plot( y=["HR"])
    df=df.fillna(df.mean(),inplace=False)
    return df

dataframe_hrv = missing_values(dataframe_hrv)

selected_x_columns = ['HR','interval in seconds','AVNN', 'RMSSD', 'pNN50', 'TP', 'ULF', 'VLF', 'LF', 'HF','LF_HF']

X = dataframe_hrv[selected_x_columns]
y = dataframe_hrv['stress']

def do_tpot(generations=5, population_size=10,X='',y=''):
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.80,test_size=0.20)
    tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2,cv=3)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_pipeline.py')
    return tpot

tpot_classifer = do_tpot(generations=100, population_size=20,X=X,y=y)
joblib.dump(tpot_classifer, 'stress_heartrate.pkl') 
