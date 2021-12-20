#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: Importez vos modules ici

import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


# TODO: Définissez vos fonctions ici


def separer_ensemble_xy(access):
    df_tot = pd.read_csv(access, delimiter=';')  # careful !! the ; dummy
    y = df_tot['quality']
    x = df_tot.drop(columns=["quality"])  # on prend tout sauf le y, pcq ils sont séparés
    # NOT x = df_tot.index !!
    return x, y


def train_split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x.to_numpy(), y.to_numpy())  # ALWAYS check for the type of ur arguments
    # (in article it says the type has to be numpy, NOT pd.DATAFRAME)
    return x_train, x_test, y_train, y_test


def AI(x_train, x_test, y_train, y_test):
    regr_forest = RandomForestRegressor()
    regr_forest.fit(x_train, y_train)
    reg_lr = LinearRegression().fit(x_train, y_train)
    return x_test, y_test, regr_forest, reg_lr


def predictions(x_test, y_test, regr_forest, reg_lr):
    pred_forest = regr_forest.predict(x_test)
    pred_linreg = reg_lr.predict(x_test)

    return x_test, pred_forest, pred_linreg, y_test


def diagrams(x_test, pred_forest, pred_linreg, y_test):
    fig = plt.figure()
    plt.plot(x_test, y_test, label="Real values")
    plt.plot(x_test, pred_linreg, label="Forest predicted value")
    fig.savefig(f"./{'Forest'}.png")
    plt.show()


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    x, y = (separer_ensemble_xy('./data/winequality-white.csv'))
    print(x,y)

    x_train, x_test, y_train, y_test = (train_split(x, y))

    x_test, y_test, regr_forest, reg_lr = AI(x_train, x_test, y_train, y_test)

    x_test, pred_forest, pred_linreg, y_test = predictions(x_test, y_test, regr_forest, reg_lr)
    print(pred_forest, pred_linreg, y_test)

    graph = diagrams(x_test, pred_forest, pred_linreg, y_test)
    print(graph)
