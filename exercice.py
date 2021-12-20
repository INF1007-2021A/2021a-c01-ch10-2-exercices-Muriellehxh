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



def separer_ensemble_xy():

    access = 'data/winequality-white.csv'
    df_tot = pd.read_csv(access, delimiter=';')  # careful !! the ; dummy

    y = df_tot['quality']
    print(y)

    x = df_tot.drop(columns=["quality"])
    # NOT x = df_tot.index !!
    print(list(x))
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print(x_train)

    regr_forest = RandomForestRegressor()
    regr_forest.fit(x_train, y_train)
    reg = LinearRegression().fit(x_train, y_train)

    pred_forest = regr_forest.predict(x_test)
    print(pred_forest)
    pred_linreg = reg.predict(x_test)
    print(pred_linreg)

    df_y_test = pd.DataFrame(y_test)
    df_y_linreg_pred = pd.DataFrame(pred_linreg)
    df_y_forest_pred = pd.DataFrame(pred_forest)
    # same number


    plt.plot(np.arange(len(y_test)), y_test, label="Real values")
    plt.plot(np.arange(len(y_test)), df_y_forest_pred, label="Forest predicted value")
    plt.show()



if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    separer_ensemble_xy()