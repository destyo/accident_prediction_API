import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from lightgbm import LGBMClassifier
import pandas as pd
from joblib import dump
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from sklearn import metrics
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def train():

    ruta = "NCDB_1999_to_2014.csv"
    df = pd.read_csv(ruta)

    new_names = ["year", "month", "weekday", "hour", "fatality", "vehicles_involved", "crash_type", "crash_place", "crash_weather",
    "surface_state", "road_slope", "traffic_state", "vehicle_id", "vehicle_type", "vehicle_year", "passenger_id", "passenger_sex",
    "passenger_age", "passenger_role", "passenger_fatality", "passenger_safety", "passenger_type"]
    df.columns = new_names

    df.drop(["passenger_id","passenger_fatality",'vehicle_id'], axis=1, inplace=True)

    df['fatality'].replace({2:0}, inplace=True)

    df.replace({"U": np.nan, "UU": np.nan, "UUUU": np.nan}, inplace=True)

    # Análisis Nulos por filas
    null_rows = df.isnull().sum(axis=1).sort_values(ascending=False)
    nulos_filas = pd.DataFrame(null_rows, columns=['nulos_filas'])  
    nulos_filas['target'] = df['fatality'].copy()
    nulos_filas['porcentaje_filas']= nulos_filas['nulos_filas']/df.shape[1]
    nulos_40 = list(nulos_filas.index[nulos_filas.porcentaje_filas>=0.40])
    df = df.loc[set(df.index)-set(nulos_40)].reset_index()

    df['vehicle_year'] = pd.to_numeric(df.year - pd.to_numeric(df.vehicle_year, errors= "coerce"))
    df.rename(columns={"vehicle_year": 'vehicle_age'}, inplace=True)

    df['passenger_age'] = pd.to_numeric(df['passenger_age'],"coerce")
    df['vehicles_involved'] = pd.to_numeric(df['vehicles_involved'],"ignore")

    df["month"] = np.int8(df["month"].replace({"01": 1, "02": 2, "11": 11, "12" : 12}))
    df = df[df["month"] != 0]

    df["weekday"] = np.int8(df["weekday"].replace({"7": 7, "1": 1, "2": 2, "3" : 3, "4":4, "5":5, "6":6}))
    df = df[df["weekday"] != 0]

    df["hour"] = pd.to_numeric(df['hour'])

    df.passenger_sex.replace({"M":1, "F":0}, inplace=True)
    df.passenger_sex.replace('[^0-9]+',np.nan,regex=True,inplace=True)
    df.passenger_sex.dropna(inplace=True)

    df = df.loc[df['passenger_safety'] != "11"]

    df = df.loc[(df['passenger_role'] == "11") | (df['passenger_role'] == "99")]
    df = df.drop('passenger_role', axis = 1)

    # Convertimos meses en cuatrimestres
    df['month'] = (df['month']-1)//3 + 1
    df = df.rename({'month': 'quarter'}, axis= 1)

    # Reducimos weekday a tres tramos
    df['weekday'] = df['weekday'].replace({6:3, 7:3, 1:2, 4:2, 5:2, 2:1, 3:1})

    # Reducimos horas a seis tramos 
    df['hour'] = df['hour'].replace({0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 11:4, 12:4, 13:4, 14:4, 15:4, 16:4, 17:4, 18:4, 19:5, 20:5, 21:5, 22:6, 23:6})


    df.drop(['index'], axis=1, inplace=True)

    # Generamos el pipeline
    # Categoricas pequeñas para OHE 
    few_cat = []
    for column in df.columns: 
        if (len(df[column].unique())< 7):
            few_cat.append(column)

    if 'fatality' in few_cat: 
        few_cat.remove('fatality')

    # Categoricas grandes para ME
    many_cat = []
    for column in df.columns[df.dtypes=='object']: 
        if (len(df[column].unique()) > 7):
            many_cat.append(column)

    # numericas
    numeric = ['vehicle_age', 'passenger_age', 'vehicles_involved', 'year']

    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])


    onehot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=np.nan)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('target', TargetEncoder(handle_unknown='ignore'))])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric),
            ('fcat', onehot_transformer, few_cat),
            ('mcat', categorical_transformer, many_cat)])

    # split

    xtrain, xtest, ytrain, ytest = train_test_split(df.drop(columns=['fatality']), df['fatality'], test_size=0.20, random_state=0)


    # save train / test
    ytest.to_csv('ytest.csv')
    xtest.to_parquet('xtest.parquet')

    # Models training
    
    lightgbm = Pipeline(steps=[('preprocessor', preprocessor),
                            ('clasificador', LGBMClassifier())])

    # param grid

    param_grid = { 
    'clasificador__is_unbalance': [True],
    'clasificador__max_depth': [-1],
    'clasificador__objective': ['binary'],
    'clasificador__learning_rate': [0.05], 
    'clasificador__n_estimators' : [300], 
    'clasificador__importance_type' : ['split'],
    'clasificador__num_leaves': [200]
    }

    CV = GridSearchCV(lightgbm, param_grid, cv=3, n_jobs=4, scoring='roc_auc', verbose=1)
    CV.fit(xtest,ytest)

    # Save model
    dump(CV, 'lightgbm.joblib')


if __name__ == '__main__':
    train()
