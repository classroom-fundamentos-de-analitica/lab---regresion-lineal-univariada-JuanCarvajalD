import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def pregunta_01():
    df = pd.read_csv('gm_2008_region.csv')
    y = df['life']
    X = df['fertility']
    print(y.shape)
    print(X.shape)
    y_reshaped = y.values.reshape(-1, 1)
    X_reshaped = X.values.reshape(-1, 1)
    print(y_reshaped.shape)
    print(X_reshaped.shape)

def pregunta_02():
    df = pd.read_csv('gm_2008_region.csv')
    print(df.shape)
    print('{:.4f}'.format(df['life'].corr(df['fertility'])))
    print('{:.4f}'.format(df['life'].mean()))
    print(type(df['fertility']))
    print('{:.4f}'.format(df['GDP'].corr(df['life'])))

def pregunta_03():
    df = pd.read_csv('gm_2008_region.csv')
    X_fertility = df['fertility'].values.reshape(-1, 1)
    y_life = df['life'].values.reshape(-1, 1)
    reg = LinearRegression()
    prediction_space = np.linspace(np.min(X_fertility), np.max(X_fertility)).reshape(-1, 1)
    reg.fit(X_fertility, y_life)
    y_pred = reg.predict(prediction_space)
    print(reg.score(X_fertility, y_life).round(4))

def pregunta_04():
    df = pd.read_csv('gm_2008_region.csv')
    X_fertility = df['fertility'].values.reshape(-1, 1)
    y_life = df['life'].values.reshape(-1, 1)
    (X_train, X_test, y_train, y_test) = train_test_split(X_fertility, y_life, test_size=0.2, random_state=53)
    linearRegression = LinearRegression()
    linearRegression.fit(X_train, y_train)
    y_pred = linearRegression.predict(X_test)
    print("R^2: {:6.4f}".format(linearRegression.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))

