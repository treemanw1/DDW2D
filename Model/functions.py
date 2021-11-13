import pandas as pd
import numpy as np


# return cost given line(theta) + points(df)
def cost_function(theta, df):
    # theta.shape = (6, 1), type(theta) = pd.DataFrame
    if len(theta) != 6:
        return "Number of features wrong la bodo"
    m = len(df)
    features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']
    cost = (1/(2*m))*(df[features].dot(theta) - df['Deaths']).apply(lambda x: x**2).sum(axis=1)
    return cost


def gradient_desc(theta, df, alpha, n):
    # theta: initial theta values, theta.shape = (6, 1), type(theta) = pd.DataFrame
    # alpha: step size
    # n: number of iterations
    m = len(df)
    for x in range(n):
        theta = theta - alpha*(1/m)
