import pandas as pd
import numpy as np


def pearsons_r(df, feature, label):
    x = df[feature].to_numpy()
    y = df[label].to_numpy()
    num = ((x - x.mean())*(y - y.mean())).sum()
    x1 = ((x - x.mean())**2).sum()
    y1 = ((y - y.mean())**2).sum()
    den = (x1*y1)**0.5
    return num/den
